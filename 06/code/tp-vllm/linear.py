"""
Tensor Parallel Linear Layers
vLLM-style implementation with proper weight loading

This module implements:
- ColumnParallelLinear: Splits weight matrix along columns (output dimension)
- RowParallelLinear: Splits weight matrix along rows (input dimension)
- QKVParallelLinear: Fused QKV projection with proper head sharding

Key differences from generic TP:
- Proper weight loading that shards weights when loading from checkpoint
- Support for fused QKV layers
- Inference-optimized (no autograd, proper dtype handling)
- Matches vLLM's weight_loader pattern
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from typing import Optional

# Try relative import first (when used as package), fallback to absolute (when used as script)
try:
    from .parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_all_gather,
    )
except ImportError:
    from parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_all_gather,
    )


def divide(numerator: int, denominator: int) -> int:
    """Divide numerator by denominator, ensuring integer result"""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism (vLLM-style).
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension (columns/output dimension) as A = [A_1, ..., A_p].
    
    Key features:
    - Proper weight loading that shards weights from checkpoint
    - Inference-optimized (no autograd overhead)
    - Matches vLLM's ColumnParallelLinear API
    
    Args:
        input_size: First dimension of matrix A (input features)
        output_size: Second dimension of matrix A (output features)
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        params_dtype: Data type for parameters (default: float32)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Get tensor parallel info
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        
        # Divide the weight matrix along the last dimension (columns)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.gather_output = gather_output
        self.params_dtype = params_dtype or torch.float32
        
        # Create weight matrix shard: [output_size_per_partition, input_size]
        # F.linear expects weight as [out_features, in_features]
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition, dtype=self.params_dtype)
        )
        
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)
        
        # Initialize weights (will be overwritten by weight_loader if loading from checkpoint)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def weight_loader(self, loaded_weight: torch.Tensor):
        """
        Load and shard weight from checkpoint (vLLM-style).
        
        This method shards the weight along the output dimension (column-wise)
        and loads only the shard for this TP rank.
        
        Args:
            loaded_weight: Full weight tensor from checkpoint [output_size, input_size]
        """
        # Shard along output dimension (dimension 0 for weight matrix)
        output_dim = 0
        shard_size = self.weight.shape[output_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight_shard = loaded_weight.narrow(output_dim, start_idx, shard_size)
        
        assert self.weight.shape == loaded_weight_shard.shape, \
            f"Weight shape mismatch: {self.weight.shape} vs {loaded_weight_shard.shape}"
        self.weight.data.copy_(loaded_weight_shard)
    
    def bias_loader(self, loaded_bias: torch.Tensor):
        """
        Load and shard bias from checkpoint.
        
        Args:
            loaded_bias: Full bias tensor from checkpoint [output_size]
        """
        if self.bias is not None:
            shard_size = self.bias.shape[0]
            start_idx = self.tp_rank * shard_size
            loaded_bias_shard = loaded_bias.narrow(0, start_idx, shard_size)
            assert self.bias.shape == loaded_bias_shard.shape
            self.bias.data.copy_(loaded_bias_shard)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = XA + b
        
        Args:
            input_: Input tensor of shape [..., input_size]
        
        Returns:
            Output tensor of shape [..., output_size_per_partition] if gather_output=False
            or [..., output_size] if gather_output=True
        """
        # Matrix multiply: F.linear does input_ @ weight.t()
        # input_: [..., input_size]
        # weight: [output_size_per_partition, input_size]
        # output_parallel: [..., output_size_per_partition]
        # Note: inference_mode should be applied at the top level (demo/model), not here
        output_parallel = F.linear(input_, self.weight, self.bias)
        
        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions
            output = tensor_model_parallel_all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"out_features={self.output_size_per_partition} (per partition), "
            f"bias={self.bias is not None}, "
            f"tp_size={self.tp_size}, "
            f"gather_output={self.gather_output}"
        )


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism (vLLM-style).
    
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension (rows/input dimension) and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    
    Key features:
    - Proper weight loading that shards weights from checkpoint
    - Inference-optimized (no autograd overhead)
    - Matches vLLM's RowParallelLinear API
    
    Args:
        input_size: First dimension of matrix A (input features)
        output_size: Second dimension of matrix A (output features)
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split again.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        params_dtype: Data type for parameters (default: float32)
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Get tensor parallel info
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        
        # Divide the weight matrix along the first dimension (rows/input dimension)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.params_dtype = params_dtype or torch.float32
        
        # Create weight matrix shard: [output_size, input_size_per_partition]
        # F.linear expects weight as [out_features, in_features]
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition, dtype=self.params_dtype)
        )
        
        if bias:
            # Bias is not parallelized, so it has full output_size
            self.bias = Parameter(torch.empty(self.output_size, dtype=self.params_dtype))
        else:
            self.register_parameter("bias", None)
        
        # Initialize weights (will be overwritten by weight_loader if loading from checkpoint)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def weight_loader(self, loaded_weight: torch.Tensor):
        """
        Load and shard weight from checkpoint (vLLM-style).
        
        This method shards the weight along the input dimension (row-wise)
        and loads only the shard for this TP rank.
        
        Args:
            loaded_weight: Full weight tensor from checkpoint [output_size, input_size]
        """
        # Shard along input dimension (dimension 1 for weight matrix)
        input_dim = 1
        shard_size = self.weight.shape[input_dim]
        start_idx = self.tp_rank * shard_size
        loaded_weight_shard = loaded_weight.narrow(input_dim, start_idx, shard_size)
        
        assert self.weight.shape == loaded_weight_shard.shape, \
            f"Weight shape mismatch: {self.weight.shape} vs {loaded_weight_shard.shape}"
        self.weight.data.copy_(loaded_weight_shard)
    
    def bias_loader(self, loaded_bias: torch.Tensor):
        """
        Load bias from checkpoint (bias is not sharded).
        
        Args:
            loaded_bias: Full bias tensor from checkpoint [output_size]
        """
        if self.bias is not None:
            assert self.bias.shape == loaded_bias.shape
            self.bias.data.copy_(loaded_bias)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = XA + b
        
        Args:
            input_: Input tensor of shape [..., input_size_per_partition] if input_is_parallel=True
                    or [..., input_size] if input_is_parallel=False
        
        Returns:
            Output tensor of shape [..., output_size]
        """
        # Note: inference_mode should be applied at the top level (demo/model), not here
        if self.input_is_parallel:
            input_parallel = input_
        else:
            # Split input along the last dimension
            # input_: [..., input_size]
            # Split into [..., input_size_per_partition] for each GPU
            split_size = self.input_size_per_partition
            start_idx = self.tp_rank * split_size
            end_idx = start_idx + split_size
            input_parallel = input_[..., start_idx:end_idx].contiguous()
        
        # Matrix multiply: F.linear does input_parallel @ weight.t()
        # input_parallel: [..., input_size_per_partition]
        # weight: [output_size, input_size_per_partition]
        # output_parallel: [..., output_size]
        # Only add bias on rank 0 to avoid adding it multiple times
        # After all-reduce, all ranks will have the same result including bias
        bias_to_use = self.bias if (self.bias is not None and self.tp_rank == 0) else None
        output_parallel = F.linear(input_parallel, self.weight, bias_to_use)
        
        if self.reduce_results and self.tp_size > 1:
            # All-reduce to sum partial results
            # If bias was added on rank 0, it will be included in the sum
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel
            # If not reducing and bias wasn't added, add it now
            if self.bias is not None and self.tp_rank != 0:
                output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size_per_partition} (per partition), "
            f"out_features={self.output_size}, "
            f"bias={self.bias is not None}, "
            f"tp_size={self.tp_size}, "
            f"input_is_parallel={self.input_is_parallel}, "
            f"reduce_results={self.reduce_results}"
        )


# Import math for initialization
import math


class QKVParallelLinear(ColumnParallelLinear):
    """
    Linear layer for fused QKV transformation (vLLM-style).
    
    This handles the common case where Q, K, V projections are fused into
    a single weight matrix. The weight is concatenated along the output dimension:
    [Q_weights | K_weights | V_weights]
    
    Key features:
    - Handles GQA/MQA (different number of KV heads vs Q heads)
    - Proper weight loading for fused QKV checkpoints
    - Supports both fused and separate QKV formats
    
    Args:
        hidden_size: Input hidden state size
        head_size: Size of each attention head
        total_num_heads: Total number of query heads
        total_num_kv_heads: Total number of key/value heads (for GQA)
        bias: If true, add bias
        params_dtype: Data type for parameters
    """
    
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        params_dtype: Optional[torch.dtype] = None,
    ):
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        
        # Get TP info first (before calling super)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        
        # Calculate local heads per TP rank
        num_heads_local = divide(total_num_heads, tp_size)
        
        # Handle KV heads: if TP size >= num_kv_heads, replicate KV heads
        if tp_size >= total_num_kv_heads:
            num_kv_heads_local = 1
            num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            num_kv_heads_local = divide(total_num_kv_heads, tp_size)
            num_kv_head_replicas = 1
        
        # Global output size: Q + K + V (total across all ranks)
        # This is the total output size before TP sharding
        # ColumnParallelLinear will divide this by tp_size internally
        global_output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        
        # Call parent constructor with global output size
        # ColumnParallelLinear will divide by tp_size to get local output size
        super().__init__(
            input_size=hidden_size,
            output_size=global_output_size,
            bias=bias,
            gather_output=False,  # Never gather for QKV
            params_dtype=params_dtype,
        )
        
        # Now set local head counts (after super().__init__ sets tp_size)
        self.num_heads = num_heads_local
        self.num_kv_heads = num_kv_heads_local
        self.num_kv_head_replicas = num_kv_head_replicas
        
        # Store output sizes for Q, K, V separately (local per rank)
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size
        
        # Verify that the weight shape matches expected local size
        # After ColumnParallelLinear divides, we should have:
        # weight.shape[0] == (num_heads_local + 2*num_kv_heads_local) * head_size
        expected_local_output_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_size
        assert self.weight.shape[0] == expected_local_output_size, (
            f"QKVParallelLinear weight shape mismatch: "
            f"expected {expected_local_output_size}, got {self.weight.shape[0]}. "
            f"num_heads_local={self.num_heads}, num_kv_heads_local={self.num_kv_heads}, "
            f"head_size={self.head_size}"
        )
        
        # Assert that num_heads_local is divisible by num_kv_heads_local for GQA
        # This ensures repeat_interleave works correctly in attention
        assert self.num_heads % self.num_kv_heads == 0, (
            f"GQA requires num_heads_local ({self.num_heads}) to be divisible by "
            f"num_kv_heads_local ({self.num_kv_heads})"
        )
    
    def _get_shard_offset_mapping(self, shard_id: str) -> int:
        """Get offset in output dimension for Q/K/V shard"""
        mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
        }
        return mapping.get(shard_id, 0)
    
    def _get_shard_size_mapping(self, shard_id: str) -> int:
        """Get size in output dimension for Q/K/V shard"""
        mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return mapping.get(shard_id, 0)
    
    def weight_loader_qkv(
        self,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        """
        Load QKV weight from checkpoint (vLLM-style).
        
        CRITICAL: PyTorch Linear.weight has shape [out_features, in_features]
        HuggingFace checkpoint may have [in_features, out_features] (transposed).
        We need to handle both formats and shard along dim=0 (out_features).
        
        Handles both:
        - Fused QKV: single weight matrix
        - Separate Q/K/V: loaded_shard_id indicates which to load
        
        Args:
            loaded_weight: Weight tensor from checkpoint
            loaded_shard_id: "q", "k", "v", or None (for fused)
        """
        # PyTorch Linear.weight has shape [out_features, in_features]
        # HuggingFace checkpoints also use this format
        # CRITICAL: Add hard assertion to verify weight shape matches expected format
        # This prevents silent errors when loading different checkpoint formats (quantized, exported, etc.)
        if loaded_shard_id is None:
            # Fused QKV: verify total size matches expected
            total_q_size = self.total_num_heads * self.head_size
            total_kv_size = self.total_num_kv_heads * self.head_size
            expected_total_out = total_q_size + 2 * total_kv_size
            assert loaded_weight.shape[1] == self.hidden_size, (
                f"Fused QKV weight shape mismatch: expected in_features={self.hidden_size}, "
                f"got {loaded_weight.shape[1]}"
            )
            assert loaded_weight.shape[0] == expected_total_out, (
                f"Fused QKV weight shape mismatch: expected out_features={expected_total_out}, "
                f"got {loaded_weight.shape[0]}. This weight cannot be split into Q/K/V parts."
            )
        else:
            # Separate Q/K/V: verify shape matches expected for this projection
            if loaded_shard_id == "q":
                expected_out = self.total_num_heads * self.head_size
            else:  # "k" or "v"
                expected_out = self.total_num_kv_heads * self.head_size
            
            assert loaded_weight.shape[1] == self.hidden_size, (
                f"{loaded_shard_id.upper()} weight shape mismatch: expected in_features={self.hidden_size}, "
                f"got {loaded_weight.shape[1]}"
            )
            assert loaded_weight.shape[0] == expected_out, (
                f"{loaded_shard_id.upper()} weight shape mismatch: expected out_features={expected_out}, "
                f"got {loaded_weight.shape[0]}"
            )
        
        if loaded_shard_id is None:
            # Fused QKV: split and load each part
            # After transpose (if needed), format is: [(total_q + total_k + total_v) * head_size, hidden_size]
            total_q_size = self.total_num_heads * self.head_size
            total_kv_size = self.total_num_kv_heads * self.head_size
            
            # Extract Q, K, V parts along dim=0 (out_features)
            q_weight = loaded_weight[:total_q_size, :]
            k_weight = loaded_weight[total_q_size:total_q_size + total_kv_size, :]
            v_weight = loaded_weight[total_q_size + total_kv_size:total_q_size + 2 * total_kv_size, :]
            
            # Shard each part along dim=0 (out_features)
            q_shard = self._shard_weight(q_weight, "q")
            k_shard = self._shard_weight(k_weight, "k")
            v_shard = self._shard_weight(v_weight, "v")
            
            # Concatenate shards along dim=0: [shard_size, hidden_size]
            self.weight.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))
        else:
            # Separate Q/K/V: load specific shard
            assert loaded_shard_id in ["q", "k", "v"]
            shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
            shard_size = self._get_shard_size_mapping(loaded_shard_id)
            
            # Shard the loaded weight along dim=0 (out_features)
            sharded = self._shard_weight(loaded_weight, loaded_shard_id)
            
            # Place in correct position in weight matrix (along dim=0)
            self.weight.data[shard_offset:shard_offset + shard_size, :].copy_(sharded)
    
    def _shard_weight(self, weight: torch.Tensor, shard_id: str) -> torch.Tensor:
        """
        Shard weight for Q, K, or V based on TP rank.
        
        CRITICAL: PyTorch Linear.weight has shape [out_features, in_features]
        For ColumnParallelLinear, we shard along dim=0 (out_features), NOT dim=1!
        
        Args:
            weight: Weight tensor from checkpoint [out_features, in_features]
            shard_id: "q", "k", or "v"
        
        Returns:
            Sharded weight [shard_size, in_features]
        """
        if shard_id == "q":
            # Shard Q heads along dim=0 (out_features)
            shard_size = self.num_heads * self.head_size
            start_idx = self.tp_rank * shard_size
            return weight.narrow(0, start_idx, shard_size)
        else:
            # Shard K or V heads along dim=0 (out_features)
            if self.num_kv_head_replicas > 1:
                # Replicate: each rank gets same KV heads
                kv_head_idx = self.tp_rank // self.num_kv_head_replicas
                shard_size = self.num_kv_heads * self.head_size
                start_idx = kv_head_idx * shard_size
                return weight.narrow(0, start_idx, shard_size)
            else:
                # Partition: shard KV heads across ranks
                shard_size = self.num_kv_heads * self.head_size
                start_idx = self.tp_rank * shard_size
                return weight.narrow(0, start_idx, shard_size)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute QKV in one go.
        
        Args:
            input_: Input tensor [..., hidden_size]
        
        Returns:
            Concatenated QKV tensor [..., (q_size + kv_size + kv_size)]
        """
        # Use parent's forward (already handles TP sharding)
        return super().forward(input_)
    
    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split fused QKV output into Q, K, V.
        
        Args:
            qkv: Fused QKV tensor [..., (q_size + kv_size + kv_size)]
        
        Returns:
            Tuple of (q, k, v) tensors
        """
        q = qkv[..., :self.q_size]
        k = qkv[..., self.q_size:self.q_size + self.kv_size]
        v = qkv[..., self.q_size + self.kv_size:]
        return q, k, v

