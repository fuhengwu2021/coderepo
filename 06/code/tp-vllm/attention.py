"""
Tensor Parallel Attention Layer (vLLM-style)

This module implements TP-aware attention layers that work with real models.
Key features:
- QKV projection with proper head sharding
- Output projection with all-reduce
- Support for GQA/MQA
- Inference-optimized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .linear import QKVParallelLinear, RowParallelLinear
    from .parallel_state import tensor_model_parallel_all_reduce
except ImportError:
    from linear import QKVParallelLinear, RowParallelLinear
    from parallel_state import tensor_model_parallel_all_reduce


class TensorParallelAttention(nn.Module):
    """
    Tensor Parallel Multi-Head Attention (vLLM-style).
    
    This implements attention with TP following vLLM's approach:
    - QKV projection: Column parallel (heads sharded across ranks)
    - Attention computation: Local to each rank (no communication)
    - Output projection: Row parallel (all-reduce to combine)
    
    Args:
        hidden_size: Hidden dimension size
        num_heads: Total number of attention heads
        num_kv_heads: Total number of key/value heads (for GQA)
        head_dim: Dimension of each head
        bias: If true, add bias to projections
        params_dtype: Data type for parameters
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
        
        if head_dim is None:
            head_dim = hidden_size // num_heads
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        
        # QKV projection: Column parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=bias,
            params_dtype=params_dtype,
        )
        
        # Output projection: Row parallel
        self.o_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=params_dtype,
        )
        
        # Get local head counts from QKV layer
        self.num_heads_local = self.qkv_proj.num_heads
        self.num_kv_heads_local = self.qkv_proj.num_kv_heads
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through TP attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            kv_cache: Optional tuple of (k_cache, v_cache) for incremental decoding
        
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection (column parallel, already sharded)
        qkv = self.qkv_proj(hidden_states)  # [batch, seq_len, (q+k+v)_local]
        q, k, v = self.qkv_proj.split_qkv(qkv)
        
        # Reshape for attention: [batch, seq_len, num_heads_local, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads_local, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads_local, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads_local, self.head_dim)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate with cache: [batch, num_kv_heads, cached_len + seq_len, head_dim]
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        # Compute attention scores: Q @ K^T
        # q: [batch, num_heads_local, seq_len, head_dim]
        # k: [batch, num_kv_heads_local, cached_len, head_dim]
        # For GQA, we need to repeat K/V to match Q heads
        if self.num_kv_heads_local < self.num_heads_local:
            repeat_factor = self.num_heads_local // self.num_kv_heads_local
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention: [batch, num_heads_local, seq_len, cached_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed (for prefill)
        if kv_cache is None and seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=scores.dtype),
                diagonal=1
            ) * float('-inf')
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads_local, seq_len, head_dim]
        
        # Reshape for output projection: [batch, seq_len, num_heads_local * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection (row parallel, all-reduce happens inside)
        output = self.o_proj(attn_output)
        
        # Return output and updated KV cache
        return output, (k, v)
