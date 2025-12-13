"""
Demo code for One-to-all (scatter) point-to-point communication using PyTorch.

One-to-all (scatter): A one-to-all operation from a root rank can be expressed 
by merging all send and receive operations in a group. The root rank sends 
different data to all other ranks.

Usage:
    # Run with 4 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_one_to_all.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_one_to_all.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_one_to_all(rank, world_size, device):
    """
    One-to-all (scatter): Root rank sends different data to all other ranks.
    """
    print(f"\n{'='*60}")
    print(f"Demo: One-to-all (Scatter) (Rank {rank})")
    print(f"{'='*60}")
    
    root = 0
    size = 2  # Size of data each rank receives
    
    if rank == root:
        # Root rank prepares different data for each rank
        send_buffers = []
        for r in range(world_size):
            send_buff = torch.tensor([r * 10 + 1, r * 10 + 2], 
                                    dtype=torch.float32, device=device)
            send_buffers.append(send_buff)
            print(f"Rank {rank} (root) preparing to send to rank {r}: {send_buff}")
        
        # Root sends to all ranks (including itself)
        for r in range(world_size):
            if r != root:
                dist.send(send_buffers[r], dst=r)
            else:
                # Root keeps its own data
                recv_tensor = send_buffers[r].clone()
    else:
        # Other ranks receive from root
        recv_tensor = torch.zeros(size, dtype=torch.float32, device=device)
        print(f"Rank {rank} waiting to receive from root...")
        dist.recv(recv_tensor, src=root)
    
    print(f"Rank {rank} received: {recv_tensor}")
    print(f"Expected: Rank {rank} should receive [{rank}*10+1, {rank}*10+2] = [{rank*10+1}, {rank*10+2}]")

def main():
    parser = argparse.ArgumentParser(description='PyTorch One-to-all (Scatter) Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch One-to-all (Scatter) Point-to-Point Communication Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_one_to_all(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: One-to-all (Scatter) demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

