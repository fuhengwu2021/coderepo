"""
Simplified RadixCache implementation for prefix cache reuse.

This is a simplified version of SGLang's RadixCache that implements
the core functionality: storing KV cache in a radix tree and matching
prefixes to reuse cached values.
"""

from collections import defaultdict
from typing import List, Optional, Dict, Tuple
import torch
import time


class RadixKey:
    """Key for radix tree lookup - contains token IDs and optional extra key."""
    
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
    ):
        self.token_ids = token_ids
        self.extra_key = extra_key
    
    def __len__(self) -> int:
        return len(self.token_ids)
    
    def __getitem__(self, idx) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)
    
    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:
    """Node in the radix tree storing KV cache indices."""
    
    counter = 0
    
    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(lambda: None)
        self.parent: Optional[TreeNode] = None
        self.key: Optional[RadixKey] = None
        # Value stores the KV cache indices for this node's tokens
        self.value: Optional[List[torch.Tensor]] = None
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1
    
    @property
    def evicted(self):
        return self.value is None
    
    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class MatchResult:
    """Result of prefix matching operation."""
    
    def __init__(
        self,
        device_indices: torch.Tensor,
        last_device_node: TreeNode,
        last_host_node: Optional[TreeNode] = None,
    ):
        self.device_indices = device_indices  # Tensor of KV cache indices
        self.last_device_node = last_device_node
        self.last_host_node = last_host_node or last_device_node


class RadixCache:
    """
    Radix tree cache for KV cache prefix reuse.
    
    This implementation stores KV cache in a radix tree structure where
    common prefixes are shared across requests, enabling efficient cache reuse.
    """
    
    def __init__(self, device: str = "cuda", page_size: int = 1):
        """
        Initialize RadixCache.
        
        Args:
            device: Device to store tensors on
            page_size: Page size for alignment (1 = token-level)
        """
        self.device = torch.device(device)
        self.page_size = page_size
        self.reset()
    
    def reset(self):
        """Reset the cache to empty state."""
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
    
    def _key_match(self, key0: RadixKey, key1: RadixKey) -> int:
        """Find the length of matching prefix between two keys."""
        if key0.extra_key != key1.extra_key:
            return 0
        
        i = 0
        min_len = min(len(key0), len(key1))
        for k0, k1 in zip(key0.token_ids[:min_len], key1.token_ids[:min_len]):
            if k0 != k1:
                break
            i += 1
        return i
    
    def _get_child_key(self, key: RadixKey):
        """Get the child key for indexing into children dict."""
        if len(key) == 0:
            return None
        if self.page_size == 1:
            return key.token_ids[0]
        else:
            return tuple(key.token_ids[:self.page_size])
    
    def match_prefix(self, key: RadixKey) -> MatchResult:
        """
        Find the longest cached prefix of key in the radix tree.
        
        Args:
            key: RadixKey to match
            
        Returns:
            MatchResult with device_indices (KV cache indices) and last_node
        """
        if len(key) == 0:
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self.root_node,
            )
        
        # Align to page size if needed
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
        
        if len(key) == 0:
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self.root_node,
            )
        
        # Traverse tree to find longest matching prefix
        current_node = self.root_node
        matched_indices = []
        remaining_key = key
        
        while len(remaining_key) > 0:
            child_key = self._get_child_key(remaining_key)
            if child_key is None:
                break
            
            # Check if child exists
            if child_key not in current_node.children or current_node.children[child_key] is None:
                break
            
            child_node = current_node.children[child_key]
            
            # Check how much of the child's key matches
            match_len = self._key_match(child_node.key, remaining_key)
            if match_len == 0:
                break
            
            # If we match the entire child node
            if match_len == len(child_node.key):
                # Add all indices from this node
                if child_node.value is not None:
                    matched_indices.extend(child_node.value)
                
                # Update access time
                child_node.last_access_time = time.monotonic()
                child_node.hit_count += 1
                
                # Move to child and continue
                current_node = child_node
                remaining_key = remaining_key[match_len:]
            else:
                # Partial match - we need to split the node
                # For simplicity, we'll just use what we matched so far
                if child_node.value is not None:
                    matched_indices.extend(child_node.value[:match_len])
                break
        
        # Convert list of tensors to single tensor
        if matched_indices:
            device_indices = torch.cat(matched_indices)
        else:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        
        return MatchResult(
            device_indices=device_indices,
            last_device_node=current_node,
        )
    
    def insert(self, key: RadixKey, value: torch.Tensor, priority: int = 0) -> int:
        """
        Insert a key-value pair into the radix tree.
        
        Args:
            key: RadixKey to insert
            value: Tensor of KV cache indices (or token IDs if simulating)
            priority: Priority for eviction (not used in simplified version)
            
        Returns:
            Number of tokens inserted
        """
        if len(key) == 0:
            return 0
        
        # Align to page size if needed
        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]
            value = value[:page_aligned_len]
        
        if len(key) == 0:
            return 0
        
        # First, try to match prefix to see where to start inserting
        match_result = self.match_prefix(key)
        last_node = match_result.last_device_node
        matched_len = len(match_result.device_indices)
        
        # If we matched everything, nothing to insert
        if matched_len >= len(key):
            return len(key)
        
        # Insert remaining tokens
        remaining_key = key[matched_len:]
        remaining_value = value[matched_len:]
        
        current_node = last_node
        remaining_tokens = remaining_key.token_ids
        remaining_values = remaining_value
        
        # Split value into individual tokens for storage
        if len(remaining_values.shape) == 0:
            remaining_values = remaining_values.unsqueeze(0)
        
        # Insert tokens one by one (or page by page)
        idx = 0
        while idx < len(remaining_tokens):
            if self.page_size == 1:
                token = remaining_tokens[idx]
                val = remaining_values[idx] if idx < len(remaining_values) else torch.tensor([token], device=self.device)
                child_key = token
            else:
                # Page-level insertion
                page_tokens = remaining_tokens[idx:idx+self.page_size]
                page_values = remaining_values[idx:idx+self.page_size] if idx+self.page_size <= len(remaining_values) else remaining_values[idx:]
                child_key = tuple(page_tokens)
                val = page_values
            
            # Create or get child node
            if child_key not in current_node.children or current_node.children[child_key] is None:
                child_node = TreeNode()
                child_node.parent = current_node
                if self.page_size == 1:
                    child_node.key = RadixKey([token], key.extra_key)
                    child_node.value = [val.unsqueeze(0) if val.dim() == 0 else val]
                else:
                    child_node.key = RadixKey(list(page_tokens), key.extra_key)
                    child_node.value = [val]
                current_node.children[child_key] = child_node
            else:
                child_node = current_node.children[child_key]
            
            current_node = child_node
            idx += self.page_size if self.page_size > 1 else 1
        
        return len(key)
