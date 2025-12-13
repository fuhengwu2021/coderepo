"""
Demo code for Sendrecv point-to-point communication using PyTorch distributed communication.

Sendrecv: In MPI terms, a sendrecv operation is when two ranks exchange data, 
both sending and receiving at the same time. This can be done by merging both 
send and recv calls into one group.

Usage:
    # Run with 2 processes (minimum 2 for sendrecv)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_sendrecv.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_sendrecv.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_sendrecv(rank, world_size, device):
    """
    Sendrecv: Two ranks exchange data, both sending and receiving at the same time.
    """
    print(f"\n{'='*60}")
    print(f"Demo: Sendrecv (Rank {rank})")
    print(f"{'='*60}")
    
    if world_size < 2:
        print("Warning: Sendrecv requires at least 2 processes. Skipping demo.")
        return
    
    # Pair up ranks: rank 0 with rank 1, rank 2 with rank 3, etc.
    if rank % 2 == 0:
        peer = rank + 1
        if peer >= world_size:
            print(f"Rank {rank} has no peer (odd number of ranks). Skipping.")
            return
    else:
        peer = rank - 1
    
    # Each rank prepares data to send
    send_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2, rank * 10 + 3], 
                               dtype=torch.float32, device=device)
    recv_tensor = torch.zeros(3, dtype=torch.float32, device=device)
    
    print(f"Rank {rank} sending to rank {peer}: {send_tensor}")
    print(f"Rank {rank} receiving from rank {peer} (before): {recv_tensor}")
    
    # Sendrecv: both send and receive in the same group
    # In PyTorch, we use send/recv which are blocking, so we can call them sequentially
    # For true concurrent sendrecv, we would use async operations or threads
    if rank < peer:
        # Lower rank sends first, then receives
        dist.send(send_tensor, dst=peer)
        dist.recv(recv_tensor, src=peer)
    else:
        # Higher rank receives first, then sends
        dist.recv(recv_tensor, src=peer)
        dist.send(send_tensor, dst=peer)
    
    print(f"Rank {rank} received from rank {peer}: {recv_tensor}")
    print(f"Expected: Rank {rank} should receive [peer*10+1, peer*10+2, peer*10+3] from rank {peer}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Sendrecv Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch Sendrecv Point-to-Point Communication Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_sendrecv(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Sendrecv demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

