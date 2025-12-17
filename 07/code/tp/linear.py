"""
Tensor Parallel Linear Layers for SGLang-style TP.

This module implements:
- ColumnParallelLinear: Splits weight matrix along columns (for QKV projection)
- RowParallelLinear: Splits weight matrix along rows (for output projection)

Following SGLang's approach where TP is applied to attention layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    Linear layer with column parallelism.
    
    Used for QKV projection in attention layers.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension (columns) as A = [A_1, ..., A_p].
    
    Args:
        input_size: First dimension of matrix A (input features)
        output_size: Second dimension of matrix A (output features)
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
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
        
        # Create weight matrix shard: [output_size_per_partition, input_size]
        # F.linear expects weight as [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )
        else:
            self.register_parameter("bias", None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
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
    Linear layer with row parallelism.
    
    Used for output projection in attention layers.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension (rows) and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    
    Args:
        input_size: First dimension of matrix A (input features)
        output_size: Second dimension of matrix A (output features)
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split again.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
    ):
        super().__init__()
        
        # Get tensor parallel info
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        
        # Divide the weight matrix along the first dimension (rows)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        
        # Create weight matrix shard: [output_size, input_size_per_partition]
        # F.linear expects weight as [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, self.input_size_per_partition)
        )
        
        if bias:
            # Bias is not parallelized, so it has full output_size
            self.bias = nn.Parameter(torch.empty(self.output_size))
        else:
            self.register_parameter("bias", None)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Y = XA + b
        
        Args:
            input_: Input tensor of shape [..., input_size_per_partition] if input_is_parallel=True
                    or [..., input_size] if input_is_parallel=False
        
        Returns:
            Output tensor of shape [..., output_size]
        """
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
