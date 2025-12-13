"""
Demo of Ring AllReduce algorithm using PyTorch.

Ring AllReduce is a communication-efficient algorithm that:
1. Reduces data in a ring topology
2. Each rank sends data to its next neighbor and receives from its previous neighbor
3. After (world_size - 1) steps, all ranks have the reduced result

Algorithm steps:
- Phase 1 (Scatter-Reduce): Data is reduced in chunks as it circulates the ring
- Phase 2 (AllGather): Reduced chunks are distributed to all ranks

Usage:
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_ring_allreduce.py
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print


def ring_allreduce(tensor, op=dist.ReduceOp.SUM):
    """
    Implement Ring AllReduce algorithm manually.
    
    Ring AllReduce works in two phases:
    1. Scatter-Reduce: Data circulates the ring, each rank reduces chunks as they pass
    2. AllGather: Reduced chunks are distributed to all ranks
    
    Args:
        tensor: Input tensor to reduce (will be modified in-place)
        op: Reduction operation (SUM, MAX, MIN, etc.)
    
    Returns:
        The reduced tensor (same as input, modified in-place)
    """
    if not dist.is_initialized():
        return tensor
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return tensor
    
    # Flatten tensor for easier chunking
    flat_tensor = tensor.flatten()
    total_elements = flat_tensor.numel()
    
    # Split into world_size chunks
    chunk_size = (total_elements + world_size - 1) // world_size  # Ceiling division
    chunks = []
    
    # Create chunks (pad last chunk if needed)
    for i in range(world_size):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_elements)
        if start_idx < total_elements:
            chunk = flat_tensor[start_idx:end_idx].clone()
            # Pad to chunk_size if needed
            if chunk.numel() < chunk_size:
                padding = torch.zeros(chunk_size - chunk.numel(), 
                                     dtype=tensor.dtype, device=tensor.device)
                chunk = torch.cat([chunk, padding])
            chunks.append(chunk)
        else:
            chunks.append(torch.zeros(chunk_size, dtype=tensor.dtype, device=tensor.device))
    
    # Phase 1: Scatter-Reduce
    # Each rank reduces chunks as they circulate the ring
    for step in range(world_size - 1):
        # Calculate which chunk we're working on in this step
        chunk_idx = (rank - step + world_size) % world_size
        
        # Ring neighbors
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size
        
        # Send current chunk to next rank, receive from previous rank
        send_chunk = chunks[chunk_idx].clone()
        recv_chunk = torch.zeros_like(chunks[chunk_idx])
        
        # Use async operations for better performance
        send_op = dist.isend(send_chunk, dst=next_rank)
        recv_op = dist.irecv(recv_chunk, src=prev_rank)
        
        send_op.wait()
        recv_op.wait()
        
        # Reduce: combine received chunk with our chunk
        if op == dist.ReduceOp.SUM:
            chunks[chunk_idx] += recv_chunk
        elif op == dist.ReduceOp.MAX:
            chunks[chunk_idx] = torch.maximum(chunks[chunk_idx], recv_chunk)
        elif op == dist.ReduceOp.MIN:
            chunks[chunk_idx] = torch.minimum(chunks[chunk_idx], recv_chunk)
        elif op == dist.ReduceOp.PRODUCT:
            chunks[chunk_idx] *= recv_chunk
        else:
            chunks[chunk_idx] += recv_chunk  # Default to SUM
    
    # Phase 2: AllGather
    # Distribute reduced chunks to all ranks
    for step in range(world_size - 1):
        chunk_idx = (rank - step - 1 + world_size) % world_size
        
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size
        
        send_chunk = chunks[chunk_idx].clone()
        recv_chunk = torch.zeros_like(chunks[chunk_idx])
        
        # Send to next, receive from prev
        send_op = dist.isend(send_chunk, dst=next_rank)
        recv_op = dist.irecv(recv_chunk, src=prev_rank)
        
        send_op.wait()
        recv_op.wait()
        
        # Update chunk with received data
        chunks[chunk_idx] = recv_chunk
    
    # Reconstruct tensor from chunks
    result_chunks = []
    for i in range(world_size):
        chunk = chunks[i]
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_elements)
        if start_idx < total_elements:
            actual_size = end_idx - start_idx
            result_chunks.append(chunk[:actual_size])
    
    if result_chunks:
        result = torch.cat(result_chunks)
        # Reshape to original shape
        tensor.copy_(result.reshape(tensor.shape))
    
    return tensor


def demo_ring_allreduce(rank, world_size, device):
    """
    Demo Ring AllReduce algorithm.
    """
    sync_print(f"\n{'='*60}", rank=rank, world_size=world_size)
    sync_print(f"Demo: Ring AllReduce (Rank {rank})", rank=rank, world_size=world_size)
    sync_print(f"{'='*60}", rank=rank, world_size=world_size)
    
    # Each rank has different input data
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    original_tensor = tensor.clone()
    sync_print(f"Rank {rank} input: {tensor}", rank=rank, world_size=world_size)
    
    # Synchronize before starting
    if world_size > 1:
        dist.barrier()
    
    # Use Ring AllReduce
    ring_allreduce(tensor, op=dist.ReduceOp.SUM)
    
    sync_print(f"Rank {rank} after ring_allreduce(SUM): {tensor}", rank=rank, world_size=world_size)
    
    # Verify result matches PyTorch's built-in all_reduce
    expected = original_tensor.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM)
    
    if torch.allclose(tensor, expected):
        sync_print(f"Rank {rank}: ✓ Result matches PyTorch all_reduce!", rank=rank, world_size=world_size)
    else:
        sync_print(f"Rank {rank}: ✗ Mismatch! Expected: {expected}, Got: {tensor}", 
                  rank=rank, world_size=world_size)
    
    # Test with MAX operation
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    ring_allreduce(tensor, op=dist.ReduceOp.MAX)
    sync_print(f"Rank {rank} after ring_allreduce(MAX): {tensor}", rank=rank, world_size=world_size)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Ring AllReduce Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    sync_print(f"\n{'#'*60}", rank=rank, world_size=world_size)
    sync_print(f"PyTorch Ring AllReduce Algorithm Demo", rank=rank, world_size=world_size)
    sync_print(f"Rank: {rank}, World Size: {world_size}, Device: {device}", rank=rank, world_size=world_size)
    sync_print(f"{'#'*60}", rank=rank, world_size=world_size)
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_ring_allreduce(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        # Use regular print for final messages
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Ring AllReduce demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        sync_print(f"Error in rank {rank}: {e}", rank=rank, world_size=world_size)
        raise
    
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

