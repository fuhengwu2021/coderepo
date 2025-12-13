"""
Demo code for Reduce collective operation using PyTorch distributed communication.

Reduce: Performs the same operation as AllReduce, but stores the result only 
in the receive buffer of a specified root rank.

Important note: The root argument is one of the ranks (not a device number), 
and is therefore impacted by a different rank to device mapping.

Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

Usage:
    # Run with 4 processes (4 GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_reduce.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_reduce.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_reduce(rank, world_size, device):
    """
    Reduce: Performs reduction across all ranks, but only root receives the result.
    """
    sync_print(f"\n{'='*60}", rank=rank, world_size=world_size)
    sync_print(f"Demo: Reduce (Rank {rank})", rank=rank, world_size=world_size)
    sync_print(f"{'='*60}", rank=rank, world_size=world_size)
    
    root = 0
    # Each rank has different input
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    sync_print(f"Rank {rank} input: {tensor}", rank=rank, world_size=world_size)
    
    # Reduce to root rank
    dist.reduce(tensor, dst=root, op=dist.ReduceOp.SUM)
    
    if rank == root:
        sync_print(f"Rank {rank} (root) after reduce: {tensor}", rank=rank, world_size=world_size)
        sync_print(f"Expected sum: {sum(range(1, world_size + 1))} for each element", rank=rank, world_size=world_size)
    else:
        sync_print(f"Rank {rank} after reduce (unchanged): {tensor}", rank=rank, world_size=world_size)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Reduce Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    sync_print(f"\n{'#'*60}", rank=rank, world_size=world_size)
    sync_print(f"PyTorch Reduce Collective Operation Demo", rank=rank, world_size=world_size)
    sync_print(f"Rank: {rank}, World Size: {world_size}, Device: {device}", rank=rank, world_size=world_size)
    sync_print(f"{'#'*60}", rank=rank, world_size=world_size)
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_reduce(rank, world_size, device)
        
        # Use regular print for final messages to avoid barrier deadlock
        # All ranks have completed, so interleaving is acceptable
        if world_size > 1:
            dist.barrier()  # Sync before final prints
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Reduce demo completed successfully!")
        print(f"{'#'*60}\n")
        
        # Final barrier to ensure all ranks complete before destroying process group
        if world_size > 1:
            dist.barrier()
        
    except Exception as e:
        sync_print(f"Error in rank {rank}: {e}", rank=rank, world_size=world_size)
        raise
    
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

