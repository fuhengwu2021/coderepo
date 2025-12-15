"""
PagedAttention v3 implementation with Continuous Batching support.

This version implements batch attention computation for multiple sequences
using ragged batching (flattened tokens + metadata) without padding.
"""

import torch
from typing import List, Optional, Tuple
from .block_manager import BlockManager, BlockTable, Block
from .paged_attention_v2 import PagedAttentionV2


class PagedAttentionV3(PagedAttentionV2):
    """
    PagedAttention v3: Block-streaming attention with continuous batching.
    
    Extends v2 with batch processing capabilities:
    - Ragged batching: flattened tokens + metadata (no padding)
    - Batch attention computation for multiple sequences
    - Efficient block-based KV cache management
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
        Initialize PagedAttention v3.
        
        Args:
            block_size: Number of tokens per block
            num_heads: Number of query attention heads (Hq)
            head_dim: Dimension of each attention head
            max_blocks: Maximum number of blocks to pre-allocate
            device: Device to use ('cuda' or 'cpu')
            use_online_softmax: If True, use single-pass online softmax (default).
                                If False, use two-pass safe_softmax algorithm.
            num_kv_heads: Number of key/value heads (Hkv). If None, equals num_heads (MHA).
        """
        super().__init__(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_blocks=max_blocks,
            device=device,
            use_online_softmax=use_online_softmax,
            num_kv_heads=num_kv_heads
        )
    
    def compute_attention_batch(
        self,
        seq_ids: List[int],
        q_batch: torch.Tensor,
        positions: List[int],
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute attention for a batch of sequences (pseudo-batching).
        
        NOTE: This is NOT true ragged batching. It processes sequences sequentially
        using Python loops. True ragged batching would:
        - Flatten all tokens to T×D (T = total tokens across all sequences)
        - Use metadata (seq_id, position, slot_mapping) to identify tokens
        - Process all tokens in a single kernel call
        
        This implementation provides:
        - Sequential batch processing (one sequence at a time)
        - Shared block pool across sequences
        - Scheduler-based sequence management
        
        Args:
            seq_ids: List of sequence IDs (length = batch_size)
            q_batch: Query tensor of shape [batch_size, num_heads, head_dim]
            positions: List of position indices for each sequence
            scale: Scaling factor for attention scores
            
        Returns:
            Attention outputs of shape [batch_size, num_heads, head_dim]
        """
        if scale is None:
            scale = 1.0 / (self.head_dim ** 0.5)
        
        batch_size = len(seq_ids)
        outputs = []
        
        # Process each sequence sequentially (pseudo-batching)
        # TODO: Implement true ragged batching with flattened tokens + metadata
        for i in range(batch_size):
            seq_id = seq_ids[i]
            q = q_batch[i]  # [num_heads, head_dim]
            
            # Use parent class's compute_attention (supports GQA via reshape+broadcast)
            output = self.compute_attention(seq_id, q, scale)
            outputs.append(output)
        
        return torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]
    
    def append_kv_batch(
        self,
        seq_ids: List[int],
        k_batch: torch.Tensor,
        v_batch: torch.Tensor,
        positions: List[int]
    ):
        """
        Append KV cache for a batch of sequences.
        
        Args:
            seq_ids: List of sequence IDs
            k_batch: Key tensor of shape [batch_size, num_kv_heads, head_dim]
            v_batch: Value tensor of shape [batch_size, num_kv_heads, head_dim]
            positions: List of position indices for each sequence
        """
        batch_size = len(seq_ids)
        
        for i in range(batch_size):
            seq_id = seq_ids[i]
            k = k_batch[i]  # [num_kv_heads, head_dim]
            v = v_batch[i]  # [num_kv_heads, head_dim]
            token_idx = positions[i]
            
            self.append_kv(seq_id, k, v, token_idx)
    
    def get_slot_mapping(
        self,
        seq_ids: List[int],
        positions: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Get slot mapping for a batch of sequences.
        
        Maps logical (seq_id, position) to physical (block_id, offset).
        
        Args:
            seq_ids: List of sequence IDs
            positions: List of position indices
            
        Returns:
            List of (block_id, offset) tuples
        """
        slot_mapping = []
        
        for seq_id, position in zip(seq_ids, positions):
            block_table = self.block_manager.get_block_table(seq_id)
            if block_table is None:
                slot_mapping.append((0, 0))  # Invalid mapping
                continue
            
            # Calculate which block and offset
            block_index = position // self.block_manager.block_size
            offset = position % self.block_manager.block_size
            
            if block_index < len(block_table.blocks):
                block = block_table.blocks[block_index]
                slot_mapping.append((block.block_id, offset))
            else:
                slot_mapping.append((0, 0))  # Invalid mapping
        
        return slot_mapping
