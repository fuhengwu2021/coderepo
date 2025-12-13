"""
Demo code for All-to-all point-to-point communication using PyTorch.

All-to-all: An all-to-all operation would be a merged loop of send/recv operations 
to/from all peers. Each rank sends different data to every other rank and receives 
different data from every other rank.

Usage:
    # Run with 4 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_all_to_all.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_all_to_all.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_all_to_all(rank, world_size, device):
    """
    All-to-all: Each rank sends different data to every other rank and receives from all.
    """
    print(f"\n{'='*60}")
    print(f"Demo: All-to-all (Rank {rank})")
    print(f"{'='*60}")
    
    size = 2  # Size of data sent to each peer
    
    # Each rank prepares data to send to each other rank
    send_buffers = []
    for dst_rank in range(world_size):
        send_buff = torch.tensor([rank * 100 + dst_rank * 10 + 1, rank * 100 + dst_rank * 10 + 2], 
                                dtype=torch.float32, device=device)
        send_buffers.append(send_buff)
    
    send_tensor = torch.cat(send_buffers)
    print(f"Rank {rank} sending to all ranks: {send_tensor}")
    
    # Each rank receives data from all ranks
    recv_buffers = []
    for src_rank in range(world_size):
        recv_buff = torch.zeros(size, dtype=torch.float32, device=device)
        recv_buffers.append(recv_buff)
    
    # All-to-all communication
    # We need to coordinate sends and receives to avoid deadlock
    # Strategy: for each pair, lower rank sends first, higher rank receives first
    # Keep own data
    recv_buffers[rank] = send_buffers[rank].clone()
    
    # Communicate with each other rank
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue
        
        # Coordinate to avoid deadlock: lower rank sends first, higher rank receives first
        if rank < peer_rank:
            # Lower rank: send first, then receive
            dist.send(send_buffers[peer_rank], dst=peer_rank)
            dist.recv(recv_buffers[peer_rank], src=peer_rank)
        else:
            # Higher rank: receive first, then send
            dist.recv(recv_buffers[peer_rank], src=peer_rank)
            dist.send(send_buffers[peer_rank], dst=peer_rank)
    
    # Synchronize to ensure all communications complete
    if world_size > 1:
        dist.barrier()
    
    # Concatenate all received data
    recv_tensor = torch.cat(recv_buffers)
    print(f"Rank {rank} received from all ranks: {recv_tensor}")
    print(f"Expected: receives [src_rank*100 + rank*10 + 1, src_rank*100 + rank*10 + 2] from each src_rank")

def main():
    parser = argparse.ArgumentParser(description='PyTorch All-to-all Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch All-to-all Point-to-Point Communication Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_all_to_all(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: All-to-all demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

