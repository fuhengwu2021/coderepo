"""
RadixAttention layer implementation.

This layer uses RadixCache to reuse KV cache prefixes across requests.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List
try:
    from .radix_cache import RadixCache, RadixKey, MatchResult
except ImportError:
    from radix_cache import RadixCache, RadixKey, MatchResult


class RadixAttention:
    """
    Attention layer that uses RadixCache for prefix cache reuse.
    
    This is a simplified version that works with HuggingFace models
    by intercepting attention computation and using cached KV values.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_id: int,
        device: str = "cuda",
        page_size: int = 1,
    ):
        """
        Initialize RadixAttention layer.
        
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            num_kv_heads: Number of key-value heads (for GQA)
            layer_id: Layer index
            device: Device to use
            page_size: Page size for cache (1 = token-level)
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.device = device
        self.page_size = page_size
        
        # Create radix cache for this layer
        self.radix_cache = RadixCache(device=device, page_size=page_size)
        
        # Store KV cache for sequences
        # Format: {seq_id: {'k': tensor, 'v': tensor, 'token_ids': list}}
        self.kv_caches: Dict[int, Dict] = {}
        
        # Track which token indices in the cache correspond to which positions
        # This maps (seq_id, position) -> cache_index
        self.cache_index_map: Dict[tuple, int] = {}
        self.next_cache_index = 0
    
    def _get_cache_index(self, seq_id: int, position: int) -> int:
        """Get or create cache index for a sequence position."""
        key = (seq_id, position)
        if key not in self.cache_index_map:
            self.cache_index_map[key] = self.next_cache_index
            self.next_cache_index += 1
        return self.cache_index_map[key]
    
    def match_prefix(
        self,
        seq_id: int,
        token_ids: List[int],
        extra_key: Optional[str] = None,
    ) -> MatchResult:
        """
        Match prefix for a sequence to find cached KV values.
        
        Args:
            seq_id: Sequence ID
            token_ids: Token IDs for the sequence
            extra_key: Optional extra key for namespace separation
            
        Returns:
            MatchResult with cached indices
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        return self.radix_cache.match_prefix(key)
    
    def insert_prefix(
        self,
        seq_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        extra_key: Optional[str] = None,
    ) -> int:
        """
        Insert a prefix into the cache.
        
        Args:
            seq_id: Sequence ID
            token_ids: Token IDs
            kv_indices: KV cache indices (or token IDs for simulation)
            extra_key: Optional extra key
            
        Returns:
            Number of tokens inserted
        """
        key = RadixKey(token_ids=token_ids, extra_key=extra_key)
        return self.radix_cache.insert(key, kv_indices)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_id: int,
        token_ids: List[int],
        cached_k: Optional[torch.Tensor] = None,
        cached_v: Optional[torch.Tensor] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with RadixAttention.
        
        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            seq_id: Sequence ID
            token_ids: Token IDs for this forward pass
            cached_k: Cached keys from prefix match
            cached_v: Cached values from prefix match
            use_cache: Whether to use cache
            
        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # If we have cached KV, concatenate with new KV
        if use_cache and cached_k is not None and cached_v is not None:
            # cached_k/v: [num_heads, cached_len, head_dim]
            # k/v: [batch, num_kv_heads, seq_len, head_dim]
            
            # Handle GQA: repeat K and V if needed
            if self.num_kv_heads < self.num_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)  # [batch, num_heads, seq_len, head_dim]
                v = v.repeat_interleave(repeat_factor, dim=1)
            
            # Expand cached to match batch
            cached_k_batch = cached_k.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
            cached_v_batch = cached_v.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
            
            # Concatenate
            k_full = torch.cat([cached_k_batch, k], dim=2)  # [batch, num_heads, cached_len+seq_len, head_dim]
            v_full = torch.cat([cached_v_batch, v], dim=2)
        else:
            # No cache, just use current K/V
            if self.num_kv_heads < self.num_heads:
                repeat_factor = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            k_full = k
            v_full = v
        
        # Compute attention: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale  # [batch, num_heads, seq_len, k_len]
        
        # Apply causal mask if needed (for generation)
        if scores.shape[-1] > scores.shape[-2]:
            # We have cached tokens, need causal mask
            seq_len_q = scores.shape[-2]
            seq_len_k = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=scores.device, dtype=scores.dtype),
                diagonal=seq_len_k - seq_len_q + 1
            ) * float('-inf')
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_full)  # [batch, num_heads, seq_len, head_dim]
        
        return attn_output
