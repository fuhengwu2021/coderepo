"""
Tensor Parallelism (TP) implementation for SGLang-style inference.

This module implements TP that integrates with RadixAttention and scheduler,
following SGLang's approach where:
- QKV projection uses column parallelism
- Output projection uses row parallelism  
- RadixAttention works with already-sharded Q/K/V tensors
- Scheduler coordinates TP workers
"""

from .parallel_state import (
    initialize_tensor_parallel,
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)
from .linear import ColumnParallelLinear, RowParallelLinear
from .tp_model_wrapper import TPRadixAttentionModelWrapper
from .scheduler import TPScheduler

__all__ = [
    "initialize_tensor_parallel",
    "get_tp_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_gather",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPRadixAttentionModelWrapper",
    "TPScheduler",
]
