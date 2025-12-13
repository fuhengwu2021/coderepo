"""
Demo code for All-to-one (gather) point-to-point communication using PyTorch.

All-to-one (gather): An all-to-one operation to a root rank would be implemented 
by having all ranks send their data to the root rank, which receives from all ranks.

Usage:
    # Run with 4 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_all_to_one.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_all_to_one.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_all_to_one(rank, world_size, device):
    """
    All-to-one (gather): All ranks send their data to the root rank.
    """
    print(f"\n{'='*60}")
    print(f"Demo: All-to-one (Gather) (Rank {rank})")
    print(f"{'='*60}")
    
    root = 0
    size = 2  # Size of data each rank sends
    
    # Each rank prepares data to send
    send_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], 
                              dtype=torch.float32, device=device)
    print(f"Rank {rank} sending: {send_tensor}")
    
    if rank == root:
        # Root rank receives from all ranks
        recv_buffers = []
        for r in range(world_size):
            if r == root:
                # Root keeps its own data
                recv_buff = send_tensor.clone()
            else:
                recv_buff = torch.zeros(size, dtype=torch.float32, device=device)
                dist.recv(recv_buff, src=r)
            recv_buffers.append(recv_buff)
            print(f"Rank {rank} (root) received from rank {r}: {recv_buff}")
        
        # Concatenate all received data
        all_data = torch.cat(recv_buffers)
        print(f"Rank {rank} (root) gathered all data: {all_data}")
    else:
        # Other ranks send to root
        dist.send(send_tensor, dst=root)
        print(f"Rank {rank} sent data to root")

def main():
    parser = argparse.ArgumentParser(description='PyTorch All-to-one (Gather) Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch All-to-one (Gather) Point-to-Point Communication Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_all_to_one(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: All-to-one (Gather) demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

