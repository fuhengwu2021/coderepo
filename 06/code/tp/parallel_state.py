"""
Tensor Parallelism State Management
Simplified version inspired by vLLM's parallel_state.py
"""
import torch
import torch.distributed as dist
from typing import Optional


class TensorParallelGroup:
    """Manages tensor parallel group state"""
    
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None
        self.rank_in_group: int = 0
        self.world_size: int = 1
        self.ranks: list[int] = []
    
    def initialize(self, ranks: list[int], backend: str = "nccl"):
        """Initialize the tensor parallel group"""
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
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce operation across the tensor parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """All-gather operation across the tensor parallel group"""
        if self.group is None or self.world_size == 1:
            return tensor
        
        # Get the shape of the tensor
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, tensor, group=self.group)
        
        # Concatenate along the specified dimension
        return torch.cat(tensor_list, dim=dim)


# Global tensor parallel group
_tp_group: Optional[TensorParallelGroup] = None


def initialize_tensor_parallel(tensor_parallel_size: int, backend: str = "nccl"):
    """
    Initialize tensor parallel groups.
    
    Args:
        tensor_parallel_size: Number of processes to use for tensor parallelism
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """
    global _tp_group
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if tensor_parallel_size > world_size:
        raise ValueError(
            f"tensor_parallel_size ({tensor_parallel_size}) > world_size ({world_size})"
        )
    
    if world_size % tensor_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
        )
    
    # Create tensor parallel groups
    num_tp_groups = world_size // tensor_parallel_size
    group_id = rank // tensor_parallel_size
    start_rank = group_id * tensor_parallel_size
    ranks = list(range(start_rank, start_rank + tensor_parallel_size))
    
    _tp_group = TensorParallelGroup()
    _tp_group.initialize(ranks, backend=backend)
    
    return _tp_group


def get_tp_group() -> TensorParallelGroup:
    """Get the tensor parallel group"""
    global _tp_group
    if _tp_group is None:
        raise RuntimeError("Tensor parallel group is not initialized. Call initialize_tensor_parallel() first.")
    return _tp_group


def get_tensor_model_parallel_rank() -> int:
    """Return rank for the tensor model parallel group"""
    return get_tp_group().rank_in_group


def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group"""
    return get_tp_group().world_size


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across tensor parallel group"""
    return get_tp_group().all_reduce(tensor)


def tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across tensor parallel group"""
    return get_tp_group().all_gather(tensor, dim=dim)

