"""
Data Parallelism State Management
Simplified version inspired by vLLM's parallel_state.py
"""
import torch
import torch.distributed as dist
from typing import Optional


class DataParallelGroup:
    """Manages data parallel group state"""
    
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None
        self.rank_in_group: int = 0
        self.world_size: int = 1
        self.ranks: list[int] = []
    
    def initialize(self, ranks: list[int], backend: str = "nccl"):
        """Initialize the data parallel group"""
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
        """All-gather operation across the data parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor, group=self.group)
        return torch.cat(tensor_list, dim=dim)
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast operation across the data parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        
        dist.broadcast(tensor, src=src, group=self.group)
        return tensor


# Global data parallel group
_dp_group: Optional[DataParallelGroup] = None


def initialize_data_parallel(data_parallel_size: int, backend: str = "nccl"):
    """
    Initialize data parallel groups.
    
    Args:
        data_parallel_size: Number of processes to use for data parallelism
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """
    global _dp_group
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if data_parallel_size > world_size:
        raise ValueError(
            f"data_parallel_size ({data_parallel_size}) > world_size ({world_size})"
        )
    
    if world_size % data_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by data_parallel_size ({data_parallel_size})"
        )
    
    # Create data parallel groups
    num_dp_groups = world_size // data_parallel_size
    group_id = rank // data_parallel_size
    start_rank = group_id * data_parallel_size
    ranks = list(range(start_rank, start_rank + data_parallel_size))
    
    _dp_group = DataParallelGroup()
    _dp_group.initialize(ranks, backend=backend)
    
    return _dp_group


def get_dp_group() -> DataParallelGroup:
    """Get the data parallel group"""
    global _dp_group
    if _dp_group is None:
        raise RuntimeError("Data parallel group is not initialized. Call initialize_data_parallel() first.")
    return _dp_group


def get_data_parallel_rank() -> int:
    """Return rank for the data parallel group"""
    return get_dp_group().rank_in_group


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group"""
    return get_dp_group().world_size


def data_parallel_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather the input tensor across data parallel group"""
    return get_dp_group().all_gather(tensor, dim=dim)


def data_parallel_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast the input tensor across data parallel group"""
    return get_dp_group().broadcast(tensor, src=src)

