"""
Example: Running distributed code without torchrun.

This shows how to use run_distributed() to launch distributed training
without needing torchrun.
"""

import torch
import torch.distributed as dist
from mdaisy import init_distributed, sync_print, run_distributed


def demo_worker(rank, world_size, device, local_rank):
    """Worker function that runs in each process"""
    sync_print(f"\n{'#'*60}", rank=rank, world_size=world_size)
    sync_print(f"PyTorch Demo (Rank {rank})", rank=rank, world_size=world_size)
    sync_print(f"Rank: {rank}, World Size: {world_size}, Device: {device}", rank=rank, world_size=world_size)
    sync_print(f"{'#'*60}", rank=rank, world_size=world_size)
    
    # Synchronize
    if world_size > 1:
        dist.barrier()
    
    # Example: AllReduce operation
    tensor = torch.tensor([rank + 1, rank + 2], dtype=torch.float32, device=device)
    sync_print(f"Rank {rank} input: {tensor}", rank=rank, world_size=world_size)
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    sync_print(f"Rank {rank} after all_reduce: {tensor}", rank=rank, world_size=world_size)
    
    if world_size > 1:
        dist.barrier()
    
    # Final message
    if world_size > 1:
        dist.barrier()
    print(f"\n{'#'*60}")
    print(f"Rank {rank}: Demo completed successfully!")
    print(f"{'#'*60}\n")


def main():
    """Main function - can be run directly without torchrun"""
    # Run with 4 processes
    run_distributed(demo_worker, world_size=4, use_cpu=False)


if __name__ == '__main__':
    # Set multiprocessing start method (required on some systems)
    mp = torch.multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()

