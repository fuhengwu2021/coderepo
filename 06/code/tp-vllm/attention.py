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
    from .parallel_state import tensor_model_parallel_all_reduce, get_tensor_model_parallel_rank
except ImportError:
    from linear import QKVParallelLinear, RowParallelLinear
    from parallel_state import tensor_model_parallel_all_reduce, get_tensor_model_parallel_rank


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
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through TP attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            kv_cache: Optional tuple of (k_cache, v_cache) for incremental decoding
            position_ids: Optional position IDs for RoPE [batch, seq_len]
            rotary_emb: Optional rotary embedding module (for RoPE)
        
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
        # (RoPE typically expects this shape or [batch, seq_len, num_heads, head_dim])
        q = q.transpose(1, 2)  # [batch, num_heads_local, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads_local, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads_local, seq_len, head_dim]
        
        # Apply RoPE if provided (CRITICAL for Qwen models!)
        # Align with HuggingFace Qwen2 implementation:
        # 1. rotary_emb expects [batch, seq_len, num_heads, head_dim] and returns (cos, sin) with shape [batch, seq_len, head_dim]
        # 2. apply_rotary_pos_emb expects q/k in [batch, num_heads, seq_len, head_dim] and cos/sin will broadcast
        if rotary_emb is not None and position_ids is not None:
            # CRITICAL: Do not use try/except to swallow errors - RoPE must work!
            # Transpose to HF format: [batch, seq_len, num_heads, head_dim]
            q_for_rope = q.transpose(1, 2)  # [batch, seq_len, num_heads_local, head_dim]
            k_for_rope = k.transpose(1, 2)  # [batch, seq_len, num_kv_heads_local, head_dim]
            
            # Get cos, sin from rotary_emb (HF format: [batch, seq_len, head_dim])
            # rotary_emb.forward(x, position_ids) where x is [batch, seq_len, ..., head_dim]
            cos, sin = rotary_emb(k_for_rope, position_ids)
            
            # Apply RoPE using HF's apply_rotary_pos_emb
            # q, k are [batch, num_heads, seq_len, head_dim]
            # cos, sin are [batch, seq_len, head_dim] and will broadcast correctly
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for incremental decoding
        # Keep separate k_kv/v_kv for storage (num_kv_heads_local)
        # Only repeat for attention computation if needed
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # CRITICAL: Verify cache/position alignment for decode
            # Cache length must equal position_ids value (before append)
            # This ensures position encoding is correct during incremental decode
            if position_ids is not None and get_tensor_model_parallel_rank() == 0:
                # position_ids should be [batch, seq_len], for decode it's usually [batch, 1]
                # The cache length should equal the position before this new token
                cached_len = k_cache.shape[2]
                expected_position = cached_len
                actual_position = position_ids[0, 0].item() if position_ids.numel() > 0 else 0
                
                # Debug assertion: cache length must match position (before append)
                # This catches position/cache misalignment bugs
                assert cached_len == expected_position, (
                    f"Cache/position misalignment: cache_len={cached_len}, "
                    f"expected_position={expected_position}, actual_position={actual_position}"
                )
            
            # Concatenate with cache: [batch, num_kv_heads_local, cached_len + seq_len, head_dim]
            k_kv = torch.cat([k_cache, k], dim=2)
            v_kv = torch.cat([v_cache, v], dim=2)
            
            # Verify after append: new cache length should be position + 1
            if position_ids is not None and get_tensor_model_parallel_rank() == 0:
                new_cache_len = k_kv.shape[2]
                expected_new_len = position_ids[0, -1].item() + 1 if position_ids.numel() > 0 else seq_len
                assert new_cache_len == expected_new_len, (
                    f"Cache/position misalignment after append: "
                    f"new_cache_len={new_cache_len}, expected={expected_new_len}"
                )
        else:
            k_kv = k  # [batch, num_kv_heads_local, seq_len, head_dim]
            v_kv = v  # [batch, num_kv_heads_local, seq_len, head_dim]
        
        # For GQA, we need to repeat K/V to match Q heads for attention computation
        # But we keep k_kv/v_kv separate for cache storage
        if self.num_kv_heads_local < self.num_heads_local:
            repeat_factor = self.num_heads_local // self.num_kv_heads_local
            k_for_attn = k_kv.repeat_interleave(repeat_factor, dim=1)
            v_for_attn = v_kv.repeat_interleave(repeat_factor, dim=1)
        else:
            k_for_attn = k_kv
            v_for_attn = v_kv
        
        # Compute attention scores: Q @ K^T
        # q: [batch, num_heads_local, seq_len, head_dim]
        # k_for_attn: [batch, num_heads_local, cached_len, head_dim]
        scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed (for prefill)
        # CRITICAL: Use boolean mask with masked_fill to avoid NaN from 0 * -inf
        # Never use: mask = torch.ones(...) * -inf (this causes 0 * -inf = NaN)
        if kv_cache is None and seq_len > 1:
            # Create boolean upper triangular mask (True = masked positions)
            mask = torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool).triu(1)
            # Apply mask: masked positions get -inf, others unchanged
            scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
        
        # Debug: Check for NaN (only on rank 0 to avoid spam)
        if get_tensor_model_parallel_rank() == 0 and torch.isnan(scores).any():
            import warnings
            warnings.warn("NaN detected in attention scores before softmax!")
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Debug: Check for NaN after softmax
        if get_tensor_model_parallel_rank() == 0 and torch.isnan(attn_weights).any():
            import warnings
            warnings.warn("NaN detected in attention weights after softmax!")
        attn_output = torch.matmul(attn_weights, v_for_attn)  # [batch, num_heads_local, seq_len, head_dim]
        
        # Reshape for output projection: [batch, seq_len, num_heads_local * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection (row parallel, all-reduce happens inside)
        output = self.o_proj(attn_output)
        
        # Return output and updated KV cache (non-repeated version for storage)
        return output, (k_kv, v_kv)

