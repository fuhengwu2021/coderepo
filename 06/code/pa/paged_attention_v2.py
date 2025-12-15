"""
PagedAttention v2 implementation with Online Softmax and Safe Softmax.

This version implements true block-streaming attention computation using
either single-pass online softmax or two-pass safe_softmax algorithm,
without concatenating blocks.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from .block_manager import BlockManager, BlockTable, Block


class PagedAttentionV2:
    """
    PagedAttention v2: Block-streaming attention with online/safe softmax.
    
    Instead of concatenating all blocks and computing standard attention,
    this version processes blocks one by one using either:
    - Online softmax: Single-pass algorithm with running max + rescale
    - Safe softmax: Two-pass algorithm (first pass finds global max, second computes weighted sum)
    
    Both algorithms avoid the need to materialize a large concatenated tensor.
    
    Key benefits:
    - No O(L) concatenation overhead
    - Lower memory footprint (no large intermediate tensors)
    - True block-streaming computation
    - Online softmax: More efficient (single pass, one matmul per block)
    - Safe softmax: More straightforward (two passes, easier to understand)
    """
    
    def __init__(
        self,
        block_size: int = 16,
        num_heads: int = 32,
        head_dim: int = 128,
        max_blocks: int = 1000,
        device: str = "cuda",
        use_online_softmax: bool = True,
        num_kv_heads: Optional[int] = None
    ):
        """
        Initialize PagedAttention v2.
        
        Args:
            block_size: Number of tokens per block
            num_heads: Number of query attention heads (Hq)
            head_dim: Dimension of each attention head
            max_blocks: Maximum number of blocks to pre-allocate
            device: Device to use ('cuda' or 'cpu')
            use_online_softmax: If True, use single-pass online softmax (default).
                                If False, use two-pass safe_softmax algorithm.
            num_kv_heads: Number of key/value heads (Hkv). If None, equals num_heads (MHA).
                          For GQA, set this to the actual KV heads count.
        """
        # For GQA: store KV with num_kv_heads to save memory
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_heads = num_heads
        
        # Validate GQA: Hq must be divisible by Hkv for standard GQA
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) for GQA")
        
        self.g = self.num_heads // self.num_kv_heads  # Group size for GQA
        
        self.block_manager = BlockManager(
            block_size=block_size,
            num_heads=self.num_kv_heads,  # Store with KV heads, not Q heads
            head_dim=head_dim,
            max_blocks=max_blocks,
            device=device
        )
        self.head_dim = head_dim
        self.device = device
        self.use_online_softmax = use_online_softmax
    
    def append_kv(self, seq_id: int, k: torch.Tensor, v: torch.Tensor, token_idx: int):
        """
        Append KV cache for a token.
        
        Args:
            seq_id: Sequence ID
            k: Key tensor of shape [num_kv_heads, head_dim] (for GQA) or [num_heads, head_dim]
            v: Value tensor of shape [num_kv_heads, head_dim] (for GQA) or [num_heads, head_dim]
            token_idx: Logical token index in the sequence
        """
        # Store KV with num_kv_heads (no physical repeat - GQA handled in compute_attention)
        return self.block_manager.append_kv(seq_id, k, v, token_idx)
    
    def compute_attention(
        self,
        seq_id: int,
        q: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention using either online softmax or safe_softmax algorithm.
        
        Args:
            seq_id: Sequence ID
            q: Query tensor of shape [num_heads, head_dim]
            scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output of shape [num_heads, head_dim]
        """
        if self.use_online_softmax:
            return self._compute_attention_online(seq_id, q, scale)
        else:
            return self._compute_attention_safe_softmax(seq_id, q, scale)
    
    def _compute_attention_online(
        self,
        seq_id: int,
        q: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention using single-pass online softmax (block-streaming) with GQA support.
        
        This implementation uses running max + rescale algorithm:
        - Single pass over blocks (only one matmul per block)
        - Maintains running max and rescales previous accumulations when max increases
        - Avoids the need for two passes over blocks
        - Uses reshape + broadcast for GQA (no physical KV repeat)
        
        Args:
            seq_id: Sequence ID
            q: Query tensor of shape [num_heads, head_dim]
            scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output of shape [num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        block_table = self.block_manager.get_block_table(seq_id)
        if block_table is None or len(block_table.blocks) == 0:
            # No cached KV, return zeros
            return torch.zeros_like(q)
        
        # ============================================================
        # GQA: Reshape Q for broadcast semantics (思路 A from 5.75.md)
        # ============================================================
        # Q: [num_heads, head_dim] -> [num_kv_heads, g, head_dim]
        # This allows KV to be broadcast across g dimension without physical repeat
        q_reshaped = q.view(self.num_kv_heads, self.g, self.head_dim)  # [Hkv, g, D]
        
        # ============================================================
        # Single-pass online softmax with running max + rescale
        # ============================================================
        # Initialize running statistics for each (Hkv, g) combination
        # Shape: [num_kv_heads, g] - one max per Q head group
        running_max = torch.full(
            (self.num_kv_heads, self.g), 
            float('-inf'), 
            device=self.device, 
            dtype=q.dtype
        )
        weighted_sum = torch.zeros(
            (self.num_kv_heads, self.g, self.head_dim),
            device=self.device,
            dtype=q.dtype
        )
        sum_exp = torch.zeros(
            (self.num_kv_heads, self.g),
            device=self.device,
            dtype=q.dtype
        )
        
        # Single pass: iterate over blocks once (only one matmul per block)
        for block in block_table.blocks:
            num_valid = block.num_tokens
            if num_valid == 0:
                continue
            
            # Get K and V for this block: [num_valid, num_kv_heads, head_dim]
            k_block = block.k_cache[:num_valid]  # [num_valid, Hkv, head_dim]
            v_block = block.v_cache[:num_valid]  # [num_valid, Hkv, head_dim]
            
            # Transpose for matmul: [num_kv_heads, num_valid, head_dim]
            k_block_t = k_block.transpose(0, 1)  # [Hkv, num_valid, D]
            v_block_t = v_block.transpose(0, 1)  # [Hkv, num_valid, D]
            
            # Compute scores: [Hkv, g, D] @ [Hkv, D, num_valid] = [Hkv, g, num_valid]
            # Using einsum for clarity: 'hgd,hkd->hgk'
            scores_block = torch.einsum('hgd,hkd->hgk', q_reshaped, k_block_t) * scale  # [Hkv, g, num_valid]
            
            # Find max for this block: [Hkv, g]
            block_max = torch.max(scores_block, dim=-1)[0]  # [Hkv, g]
            
            # Rescale previous accumulations if we found a new max
            should_rescale = block_max > running_max  # [Hkv, g]
            new_max = torch.maximum(running_max, block_max)  # [Hkv, g]
            
            # Rescale previous accumulations only when max actually increased
            rescale_factor = torch.where(
                should_rescale,
                torch.exp(running_max - new_max),  # exp(running_max - block_max) when should_rescale=True
                torch.ones_like(running_max)  # no change, no rescale
            )
            
            weighted_sum = weighted_sum * rescale_factor.unsqueeze(-1)  # [Hkv, g, D]
            sum_exp = sum_exp * rescale_factor  # [Hkv, g]
            
            # Update running max
            running_max = new_max
            
            # Compute exp(scores - running_max) for this block
            running_max_expanded = running_max.unsqueeze(-1)  # [Hkv, g, 1]
            exp_scores = torch.exp(scores_block - running_max_expanded)  # [Hkv, g, num_valid]
            
            # Accumulate sum_exp
            block_sum_exp = torch.sum(exp_scores, dim=-1)  # [Hkv, g]
            sum_exp = sum_exp + block_sum_exp
            
            # Accumulate weighted sum: sum(exp(scores - running_max) * V)
            # exp_scores: [Hkv, g, num_valid]
            # v_block_t: [Hkv, num_valid, D]
            # We need to broadcast: [Hkv, g, num_valid] @ [Hkv, num_valid, D] = [Hkv, g, D]
            # Using einsum: 'hgk,hkd->hgd'
            block_weighted_sum = torch.einsum('hgk,hkd->hgd', exp_scores, v_block_t)  # [Hkv, g, D]
            
            weighted_sum = weighted_sum + block_weighted_sum
        
        # Normalize: divide by sum_exp
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        sum_exp_expanded = sum_exp.unsqueeze(-1)  # [Hkv, g, 1]
        output_reshaped = weighted_sum / sum_exp_expanded  # [Hkv, g, D]
        
        # Reshape back to [num_heads, head_dim]
        output = output_reshaped.view(self.num_heads, self.head_dim)  # [Hq, D]
        
        return output
    
    def _compute_attention_safe_softmax(
        self,
        seq_id: int,
        q: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention using two-pass safe_softmax algorithm (block-streaming) with GQA support.
        
        This implementation uses the log-sum-exp two-pass algorithm:
        1. First pass: Compute global max across all blocks
        2. Second pass: Compute softmax-normalized weighted sum of values
        
        This is more straightforward but requires two passes over blocks.
        Uses reshape + broadcast for GQA (no physical KV repeat).
        
        Args:
            seq_id: Sequence ID
            q: Query tensor of shape [num_heads, head_dim]
            scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            
        Returns:
            Attention output of shape [num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        block_table = self.block_manager.get_block_table(seq_id)
        if block_table is None or len(block_table.blocks) == 0:
            # No cached KV, return zeros
            return torch.zeros_like(q)
        
        # ============================================================
        # GQA: Reshape Q for broadcast semantics (思路 A from 5.75.md)
        # ============================================================
        # Q: [num_heads, head_dim] -> [num_kv_heads, g, head_dim]
        q_reshaped = q.view(self.num_kv_heads, self.g, self.head_dim)  # [Hkv, g, D]
        
        # ============================================================
        # PASS 1: Compute global maximum across all blocks
        # ============================================================
        global_max = torch.full(
            (self.num_kv_heads, self.g), 
            float('-inf'), 
            device=self.device, 
            dtype=q.dtype
        )
        
        # Iterate over blocks to find global max
        for block in block_table.blocks:
            num_valid = block.num_tokens
            if num_valid == 0:
                continue
            
            # Get K for this block: [num_valid, num_kv_heads, head_dim]
            k_block = block.k_cache[:num_valid]  # [num_valid, Hkv, D]
            
            # Transpose for matmul: [num_kv_heads, num_valid, head_dim]
            k_block_t = k_block.transpose(0, 1)  # [Hkv, num_valid, D]
            
            # Compute scores: [Hkv, g, D] @ [Hkv, D, num_valid] = [Hkv, g, num_valid]
            scores_block = torch.einsum('hgd,hkd->hgk', q_reshaped, k_block_t) * scale  # [Hkv, g, num_valid]
            
            # Update global max
            block_max = torch.max(scores_block, dim=-1)[0]  # [Hkv, g]
            global_max = torch.maximum(global_max, block_max)
        
        # ============================================================
        # PASS 2: Compute softmax-normalized weighted sum
        # ============================================================
        weighted_sum = torch.zeros(
            (self.num_kv_heads, self.g, self.head_dim),
            device=self.device,
            dtype=q.dtype
        )
        sum_exp = torch.zeros(
            (self.num_kv_heads, self.g),
            device=self.device,
            dtype=q.dtype
        )
        
        # Iterate over blocks again to compute weighted sum
        for block in block_table.blocks:
            num_valid = block.num_tokens
            if num_valid == 0:
                continue
            
            # Get K and V for this block
            k_block = block.k_cache[:num_valid]  # [num_valid, Hkv, D]
            v_block = block.v_cache[:num_valid]  # [num_valid, Hkv, D]
            
            # Transpose for computation
            k_block_t = k_block.transpose(0, 1)  # [Hkv, num_valid, D]
            v_block_t = v_block.transpose(0, 1)  # [Hkv, num_valid, D]
            
            # Compute scores for this block
            scores_block = torch.einsum('hgd,hkd->hgk', q_reshaped, k_block_t) * scale  # [Hkv, g, num_valid]
            
            # Compute exp(scores - global_max) for numerical stability
            global_max_expanded = global_max.unsqueeze(-1)  # [Hkv, g, 1]
            exp_scores = torch.exp(scores_block - global_max_expanded)  # [Hkv, g, num_valid]
            
            # Accumulate sum_exp: sum(exp(scores - global_max))
            block_sum_exp = torch.sum(exp_scores, dim=-1)  # [Hkv, g]
            sum_exp = sum_exp + block_sum_exp
            
            # Compute weighted sum: sum(exp(scores - global_max) * V)
            # Using einsum: 'hgk,hkd->hgd'
            block_weighted_sum = torch.einsum('hgk,hkd->hgd', exp_scores, v_block_t)  # [Hkv, g, D]
            
            weighted_sum = weighted_sum + block_weighted_sum
        
        # Normalize: divide by sum(exp(scores - global_max))
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        sum_exp_expanded = sum_exp.unsqueeze(-1)  # [Hkv, g, 1]
        output_reshaped = weighted_sum / sum_exp_expanded  # [Hkv, g, D]
        
        # Reshape back to [num_heads, head_dim]
        output = output_reshaped.view(self.num_heads, self.head_dim)  # [Hq, D]
        
        return output
    
    def compute_attention_batch(
        self,
        seq_ids: List[int],
        q_batch: torch.Tensor,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention for a batch of queries using online softmax.
        
        Args:
            seq_ids: List of sequence IDs
            q_batch: Query tensor of shape [batch_size, num_heads, head_dim]
            scale: Scaling factor for attention scores
            
        Returns:
            Attention outputs of shape [batch_size, num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        batch_size = len(seq_ids)
        outputs = []
        
        for i in range(batch_size):
            output = self.compute_attention(seq_ids[i], q_batch[i], scale)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]
    
    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence."""
        self.block_manager.free_sequence(seq_id)
    
    def get_stats(self) -> dict:
        """Get statistics about block usage."""
        return self.block_manager.get_stats()
