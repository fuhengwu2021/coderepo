"""
Pipeline Parallelism State Management
Simplified version inspired by vLLM's parallel_state.py
"""
import torch
import torch.distributed as dist
from typing import Optional


class PipelineParallelGroup:
    """Manages pipeline parallel group state"""
    
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None
        self.rank_in_group: int = 0
        self.world_size: int = 1
        self.ranks: list[int] = []
        self.prev_rank: Optional[int] = None  # Previous stage rank
        self.next_rank: Optional[int] = None  # Next stage rank
    
    def initialize(self, ranks: list[int], backend: str = "nccl"):
        """Initialize the pipeline parallel group"""
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        
        self.group = dist.new_group(ranks, backend=backend)
        self.ranks = ranks
        self.world_size = len(ranks)
        
        global_rank = dist.get_rank()
        if global_rank in ranks:
            self.rank_in_group = ranks.index(global_rank)
            # Determine previous and next ranks
            if self.rank_in_group > 0:
                self.prev_rank = ranks[self.rank_in_group - 1]
            if self.rank_in_group < len(ranks) - 1:
                self.next_rank = ranks[self.rank_in_group + 1]
        else:
            self.rank_in_group = 0
            self.world_size = 1
    
    def send(self, tensor: torch.Tensor, dst: int):
        """Send tensor to next stage"""
        if self.next_rank is not None and self.group is not None:
            dist.send(tensor, dst=dst, group=self.group)
    
    def recv(self, shape: tuple, dtype: torch.dtype, device: torch.device, src: int) -> torch.Tensor:
        """Receive tensor from previous stage"""
        if self.prev_rank is not None and self.group is not None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tensor, src=src, group=self.group)
            return tensor
        else:
            raise RuntimeError("No previous rank to receive from")
    
    def is_first_stage(self) -> bool:
        """Check if this is the first stage"""
        return self.prev_rank is None
    
    def is_last_stage(self) -> bool:
        """Check if this is the last stage"""
        return self.next_rank is None


# Global pipeline parallel group
_pp_group: Optional[PipelineParallelGroup] = None


def initialize_pipeline_parallel(pipeline_parallel_size: int, backend: str = "nccl"):
    """
    Initialize pipeline parallel groups.
    
    Args:
        pipeline_parallel_size: Number of processes to use for pipeline parallelism
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """
    global _pp_group
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Handle single-process case gracefully
    if world_size == 1:
        if pipeline_parallel_size > 1:
            # Can't create a group larger than world_size, just use world_size
            pipeline_parallel_size = 1
        _pp_group = PipelineParallelGroup()
        _pp_group.rank_in_group = 0
        _pp_group.world_size = 1
        _pp_group.ranks = [0]
        _pp_group.prev_rank = None  # First and only stage
        _pp_group.next_rank = None  # Last and only stage
        _pp_group.group = None  # No group needed for single process
        return _pp_group
    
    if pipeline_parallel_size > world_size:
        raise ValueError(
            f"pipeline_parallel_size ({pipeline_parallel_size}) > world_size ({world_size})"
        )
    
    if world_size % pipeline_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by pipeline_parallel_size ({pipeline_parallel_size})"
        )
    
    # Create pipeline parallel groups
    num_pp_groups = world_size // pipeline_parallel_size
    group_id = rank // pipeline_parallel_size
    start_rank = group_id * pipeline_parallel_size
    ranks = list(range(start_rank, start_rank + pipeline_parallel_size))
    
    _pp_group = PipelineParallelGroup()
    _pp_group.initialize(ranks, backend=backend)
    
    return _pp_group


def get_pp_group() -> PipelineParallelGroup:
    """Get the pipeline parallel group"""
    global _pp_group
    if _pp_group is None:
        raise RuntimeError("Pipeline parallel group is not initialized. Call initialize_pipeline_parallel() first.")
    return _pp_group


def get_pipeline_parallel_rank() -> int:
    """Return rank for the pipeline parallel group (stage index)"""
    return get_pp_group().rank_in_group


def get_pipeline_parallel_world_size() -> int:
    """Return world size for the pipeline parallel group (number of stages)"""
    return get_pp_group().world_size


def get_prev_rank() -> Optional[int]:
    """Get the rank of the previous stage"""
    return get_pp_group().prev_rank


def get_next_rank() -> Optional[int]:
    """Get the rank of the next stage"""
    return get_pp_group().next_rank


def is_first_stage() -> bool:
    """Check if this is the first pipeline stage"""
    return get_pp_group().is_first_stage()


def is_last_stage() -> bool:
    """Check if this is the last pipeline stage"""
    return get_pp_group().is_last_stage()

