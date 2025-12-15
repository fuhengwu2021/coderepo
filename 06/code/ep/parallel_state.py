"""
Expert Parallelism State Management
Simplified version inspired by vLLM's parallel_state.py
"""
import torch
import torch.distributed as dist
from typing import Optional


class ExpertParallelGroup:
    """Manages expert parallel group state"""
    
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None
        self.rank_in_group: int = 0
        self.world_size: int = 1
        self.ranks: list[int] = []
    
    def initialize(self, ranks: list[int], backend: str = "nccl"):
        """Initialize the expert parallel group"""
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        
        self.group = dist.new_group(ranks, backend=backend)
        self.ranks = ranks
        self.world_size = len(ranks)
        
        global_rank = dist.get_rank()
        if global_rank in ranks:
            self.rank_in_group = ranks.index(global_rank)
        else:
            self.rank_in_group = 0
            self.world_size = 1
    
    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather operation across the expert parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor, group=self.group)
        return torch.cat(tensor_list, dim=dim)
    
    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Reduce-scatter operation across the expert parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        
        # Split tensor along dim
        chunk_size = tensor.shape[dim] // self.world_size
        chunks = torch.chunk(tensor, self.world_size, dim=dim)
        
        # Reduce-scatter: each rank gets sum of corresponding chunks
        output = torch.empty_like(chunks[self.rank_in_group])
        dist.reduce_scatter(output, list(chunks), op=dist.ReduceOp.SUM, group=self.group)
        return output


# Global expert parallel group
_ep_group: Optional[ExpertParallelGroup] = None


def initialize_expert_parallel(expert_parallel_size: int, backend: str = "nccl"):
    """
    Initialize expert parallel groups.
    
    Args:
        expert_parallel_size: Number of processes to use for expert parallelism
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    global _ep_group
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if expert_parallel_size > world_size:
        raise ValueError(
            f"expert_parallel_size ({expert_parallel_size}) > world_size ({world_size})"
        )
    
    if world_size % expert_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by expert_parallel_size ({expert_parallel_size})"
        )
    
    # Create expert parallel groups
    num_ep_groups = world_size // expert_parallel_size
    group_id = rank // expert_parallel_size
    start_rank = group_id * expert_parallel_size
    ranks = list(range(start_rank, start_rank + expert_parallel_size))
    
    _ep_group = ExpertParallelGroup()
    _ep_group.initialize(ranks, backend=backend)
    
    return _ep_group


def get_ep_group() -> ExpertParallelGroup:
    """Get the expert parallel group"""
    global _ep_group
    if _ep_group is None:
        raise RuntimeError("Expert parallel group is not initialized. Call initialize_expert_parallel() first.")
    return _ep_group


def get_expert_parallel_rank() -> int:
    """Return rank for the expert parallel group"""
    return get_ep_group().rank_in_group


def get_expert_parallel_world_size() -> int:
    """Return world size for the expert parallel group"""
    return get_ep_group().world_size


def expert_parallel_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather the input tensor across expert parallel group"""
    return get_ep_group().all_gather(tensor, dim=dim)


def expert_parallel_reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Reduce-scatter the input tensor across expert parallel group"""
    return get_ep_group().reduce_scatter(tensor, dim=dim)

