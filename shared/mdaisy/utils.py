"""
Shared utilities for distributed PyTorch operations.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import models
import os
import sys
import time


def sync_print(*args, rank=None, world_size=1, use_sleep=False, **kwargs):
    """
    Synchronized print that ensures output from different ranks doesn't interleave.
    
    Two modes:
    1. use_sleep=False (default): Uses simple barrier before print (fast and reliable)
    2. use_sleep=True: Uses sleep delays based on rank (not reliable, for demos only)
    
    Args:
        *args: Arguments to pass to print()
        rank: Current process rank (required if world_size > 1)
        world_size: Total number of processes
        use_sleep: If True, use sleep-based ordering instead of barriers
        **kwargs: Keyword arguments to pass to print()
    """
    if world_size <= 1:
        print(*args, **kwargs)
        return
    
    if rank is None:
        # Try to get rank from environment or distributed
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = int(os.environ.get('RANK', 0))
    
    if use_sleep:
        # Simple sleep-based approach: each rank waits a bit longer
        # Note: This is not reliable as ranks execute at different speeds
        time.sleep(rank * 0.01)  # 10ms delay per rank
        print(*args, **kwargs)
        sys.stdout.flush()
        return
    
    # Simple barrier-based approach: sync all ranks, then print in order
    # Check if process group is still initialized
    if not dist.is_initialized():
        # Fallback to regular print if process group is destroyed
        print(*args, **kwargs)
        return
    
    try:
        # Simple approach: one barrier to sync, then print in rank order with barriers
        dist.barrier()
        
        # Print in rank order - each rank waits for its turn
        for r in range(world_size):
            if r == rank:
                print(*args, **kwargs)
                sys.stdout.flush()
            # Barrier after each rank's print to ensure ordering
            if r < world_size - 1:  # Don't barrier after last rank
                dist.barrier()
    except (RuntimeError, Exception):
        # If barrier fails (e.g., process group destroyed), just print normally
        print(*args, **kwargs)


def init_distributed(use_cpu=False):
    """
    Initialize distributed process group.
    
    Works with both single-node and multi-node setups when using torchrun.
    Reads environment variables set by torchrun:
    - RANK: Global rank across all nodes
    - WORLD_SIZE: Total processes across all nodes  
    - LOCAL_RANK: Rank within the current node
    
    Args:
        use_cpu: If True, force CPU usage even if GPUs are available.
                 If False, automatically falls back to CPU if not enough GPUs.
    
    Returns:
        tuple: (rank, world_size, device, local_rank)
            - rank: Global rank of the process (across all nodes)
            - world_size: Total number of processes (across all nodes)
            - device: torch.device object (CPU or CUDA)
            - local_rank: Local rank on the current node
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Fallback for single process testing
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Check if we should use CPU: explicit flag, no CUDA, or not enough GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_cpu = use_cpu or not torch.cuda.is_available() or num_gpus < world_size
    
    if use_cpu:
        device = torch.device('cpu')
        if rank == 0 and num_gpus < world_size:
            print(f"Warning: Only {num_gpus} GPU(s) available, but {world_size} processes requested. Using CPU instead.")
    else:
        # Set device BEFORE initializing process group for NCCL
        # This ensures NCCL knows which device to use
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group even for world_size == 1 to allow distributed APIs to work
    # This is necessary when code uses dist.get_rank(), dist.is_initialized(), etc.
    if not dist.is_initialized():
        # For world_size == 1, use 'gloo' backend (nccl requires at least 2 processes)
        if world_size == 1:
            backend = 'gloo'
        else:
            backend = 'gloo' if use_cpu else 'nccl'
        
        # Specify device_id for NCCL to avoid warnings about guessing device ID
        init_kwargs = {'backend': backend}
        if not use_cpu and torch.cuda.is_available() and world_size > 1:
            init_kwargs['device_id'] = local_rank
        
        # Suppress NCCL warning about unbatched P2P ops for demos
        # This warning appears when using send/recv operations
        if backend == 'nccl' and 'TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING' not in os.environ:
            os.environ['TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING'] = 'false'
        
        dist.init_process_group(**init_kwargs)
    
    return rank, world_size, device, local_rank


def run_distributed(worker_fn, world_size, use_cpu=False, *args, **kwargs):
    """
    Run a distributed function without torchrun using multiprocessing.
    
    This is an alternative to torchrun that spawns processes manually.
    
    Args:
        worker_fn: Function to run in each process. Must accept (rank, world_size, device, local_rank, *args, **kwargs)
        world_size: Number of processes to spawn
        use_cpu: If True, force CPU usage
        *args: Additional positional arguments to pass to worker_fn
        **kwargs: Additional keyword arguments to pass to worker_fn
    
    Example:
        def my_worker(rank, world_size, device, local_rank):
            # Your distributed code here
            pass
        
        run_distributed(my_worker, world_size=4)
    """
    def _worker_wrapper(rank):
        """Wrapper that sets up environment and calls worker function"""
        # Set environment variables to mimic torchrun
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(rank)  # For single-node, local_rank = rank
        
        # Initialize distributed
        rank_val, world_size_val, device, local_rank = init_distributed(use_cpu=use_cpu)
        
        try:
            # Call the worker function
            return worker_fn(rank_val, world_size_val, device, local_rank, *args, **kwargs)
        finally:
            # Cleanup
            if world_size_val > 1 and dist.is_initialized():
                dist.destroy_process_group()
    
    # Spawn processes
    mp.spawn(_worker_wrapper, nprocs=world_size, join=True)


def get_node_info():
    """
    Get information about the current node in a multi-node setup.
    
    Returns:
        dict: Dictionary with node information:
            - node_rank: Rank of this node (if available)
            - num_nodes: Number of nodes (if available)
            - hostname: Hostname of current node
            - is_master: True if this is the master node (rank 0)
    """
    info = {
        'hostname': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('HOSTNAME', 'unknown'),
        'is_master': False,
    }
    
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        info['is_master'] = (rank == 0)
        info['global_rank'] = rank
    
    if 'LOCAL_RANK' in os.environ:
        info['local_rank'] = int(os.environ['LOCAL_RANK'])
    
    if 'WORLD_SIZE' in os.environ:
        info['world_size'] = int(os.environ['WORLD_SIZE'])
    
    # Try to get node rank from environment (set by some launchers)
    if 'NODE_RANK' in os.environ:
        info['node_rank'] = int(os.environ['NODE_RANK'])
    
    # Try to infer from LOCAL_RANK and WORLD_SIZE
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # If LOCAL_RANK resets to 0, we might be on a different node
        # This is a heuristic - not always accurate
        if local_rank == 0 and 'LOCAL_WORLD_SIZE' in os.environ:
            local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            if 'RANK' in os.environ:
                rank = int(os.environ['RANK'])
                info['node_rank'] = rank // local_world_size
    
    return info


def get_resnet18_fashionmnist(num_classes=10):
    """
    Get ResNet18 model adapted for 1-channel FashionMNIST input.
    
    This function creates a ResNet18 model from torchvision and modifies it
    to work with FashionMNIST's 1-channel grayscale images instead of the
    default 3-channel RGB images.
    
    Args:
        num_classes: Number of output classes (default: 10 for FashionMNIST)
    
    Returns:
        torch.nn.Module: ResNet18 model adapted for FashionMNIST
    """
    model = models.resnet18(weights=None)  # Use pretrained=False for random init
    # Modify first conv layer to accept 1 channel instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify last fully connected layer for specified number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet18_cifar10(num_classes=10):
    """
    Get ResNet18 model adapted for 3-channel CIFAR-10 input.
    
    This function creates a ResNet18 model from torchvision and modifies it
    to work with CIFAR-10's 32×32 RGB images. Unlike FashionMNIST, CIFAR-10
    already has 3 channels, but the images are smaller (32×32 vs 224×224),
    so we adjust the first convolutional layer and remove the maxpool layer.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
    
    Returns:
        torch.nn.Module: ResNet18 model adapted for CIFAR-10
    """
    model = models.resnet18(weights=None)  # Use pretrained=False for random init
    # CIFAR-10 has 3 channels, but images are 32×32, so we adjust the first conv layer
    # Use smaller kernel and stride to preserve spatial dimensions
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the first maxpool since CIFAR-10 images are already small (32×32)
    model.maxpool = nn.Identity()
    # Modify last fully connected layer for specified number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

