"""
Demo code for Neighbor exchange point-to-point communication using PyTorch.

Neighbor exchange: Exchanging data with neighbors in an N-dimensional space.
This is useful for stencil computations, where each rank needs to exchange 
data with its neighbors in each dimension.

Note on topology:
    - With 2 processes: Uses 1D topology (since 2 is not a perfect square)
    - With 4, 9, or 16 processes: Uses 2D grid topology (perfect squares)
    - For CPU validation, 2 processes is sufficient
    - For demonstrating 2D topology, use 4, 9, or 16 processes

Usage:
    # Run with 2 processes (uses 1D topology)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_neighbor_exchange.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_neighbor_exchange.py --use_cpu
    
    # For 2D grid topology, use 4, 9, or 16 processes (perfect squares)
    # OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_neighbor_exchange.py
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print
import math

def get_neighbors_1d(rank, world_size):
    """Get neighbors in 1D topology."""
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < world_size - 1 else None
    return prev_rank, next_rank

def get_neighbors_2d(rank, world_size):
    """Get neighbors in 2D grid topology (assumes square grid)."""
    grid_size = int(math.sqrt(world_size))
    if grid_size * grid_size != world_size:
        # Not a perfect square, fall back to 1D
        return get_neighbors_1d(rank, world_size)
    
    row = rank // grid_size
    col = rank % grid_size
    
    # Previous and next in each dimension
    prev_row = row - 1 if row > 0 else None
    next_row = row + 1 if row < grid_size - 1 else None
    prev_col = col - 1 if col > 0 else None
    next_col = col + 1 if col < grid_size - 1 else None
    
    # Convert to ranks
    prev_rank_0 = prev_row * grid_size + col if prev_row is not None else None
    next_rank_0 = next_row * grid_size + col if next_row is not None else None
    prev_rank_1 = row * grid_size + prev_col if prev_col is not None else None
    next_rank_1 = row * grid_size + next_col if next_col is not None else None
    
    return [(prev_rank_0, next_rank_0), (prev_rank_1, next_rank_1)]

def demo_neighbor_exchange(rank, world_size, device):
    """
    Neighbor exchange: Exchange data with neighbors in N-dimensional space.
    """
    print(f"\n{'='*60}")
    print(f"Demo: Neighbor Exchange (Rank {rank})")
    print(f"{'='*60}")
    
    # Determine topology (1D or 2D)
    grid_size = int(math.sqrt(world_size))
    if grid_size * grid_size == world_size:
        ndims = 2
        neighbors = get_neighbors_2d(rank, world_size)
        print(f"Rank {rank} using 2D topology")
    else:
        ndims = 1
        prev_rank, next_rank = get_neighbors_1d(rank, world_size)
        neighbors = [(prev_rank, next_rank)]
        print(f"Rank {rank} using 1D topology")
    
    size = 2  # Size of data exchanged in each dimension
    
    # Prepare send buffers for each dimension
    send_buffers = []
    for d in range(ndims):
        prev_rank, next_rank = neighbors[d]
        send_buff = torch.tensor([rank * 100 + d * 10 + 1, rank * 100 + d * 10 + 2], 
                                dtype=torch.float32, device=device)
        send_buffers.append(send_buff)
        print(f"Rank {rank} dimension {d}: prev={prev_rank}, next={next_rank}, sending={send_buff}")
    
    # Prepare receive buffers for each dimension
    recv_buffers = []
    for d in range(ndims):
        prev_rank, next_rank = neighbors[d]
        recv_from_prev = torch.zeros(size, dtype=torch.float32, device=device) if prev_rank is not None else None
        recv_from_next = torch.zeros(size, dtype=torch.float32, device=device) if next_rank is not None else None
        recv_buffers.append((recv_from_prev, recv_from_next))
    
    # Exchange data with neighbors in each dimension
    for d in range(ndims):
        prev_rank, next_rank = neighbors[d]
        
        # Send to next, receive from prev
        if next_rank is not None:
            dist.send(send_buffers[d], dst=next_rank)
            print(f"Rank {rank} sent to next (rank {next_rank}) in dimension {d}")
        
        if prev_rank is not None:
            dist.recv(recv_buffers[d][0], src=prev_rank)
            print(f"Rank {rank} received from prev (rank {prev_rank}) in dimension {d}: {recv_buffers[d][0]}")
        
        # Send to prev, receive from next (reverse direction)
        if prev_rank is not None:
            dist.send(send_buffers[d], dst=prev_rank)
            print(f"Rank {rank} sent to prev (rank {prev_rank}) in dimension {d}")
        
        if next_rank is not None:
            dist.recv(recv_buffers[d][1], src=next_rank)
            print(f"Rank {rank} received from next (rank {next_rank}) in dimension {d}: {recv_buffers[d][1]}")
    
    print(f"Rank {rank} neighbor exchange completed")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Neighbor Exchange Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch Neighbor Exchange Point-to-Point Communication Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_neighbor_exchange(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Neighbor exchange demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

