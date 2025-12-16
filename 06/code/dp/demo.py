"""
Data Parallelism Demo
Demonstrates data parallelism where each rank has a full model copy
and processes different batches of data

Run with:
    torchrun --nproc_per_node=2 demo.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist

# Import shared distributed utilities from mdaisy
try:
    from mdaisy import init_distributed
except ImportError:
    # Try to add the shared directory to path
    shared_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "resources", "coderepo", "shared"
    )
    if os.path.exists(shared_path):
        sys.path.insert(0, shared_path)
        from mdaisy import init_distributed
    else:
        raise ImportError(
            "Could not import mdaisy. Please install it or ensure resources/coderepo/shared is available.\n"
            "You can install it with: pip install -e resources/coderepo/shared"
        )

from parallel_state import initialize_data_parallel
from model import SimpleModel, SimpleTransformer


def demo_data_parallel_inference(device: torch.device, dp_size: int):
    """Demonstrate data parallel inference"""
    print("\n" + "="*60)
    print("Demo 1: Data Parallel Inference")
    print("="*60)
    
    # Create a model (each rank has a full copy)
    model = SimpleModel(input_size=128, hidden_size=256, output_size=10).to(device)
    model.eval()
    
    # Create different input data for each rank
    # In real scenarios, this would be different batches from a dataset
    batch_size = 4
    dp_rank = dist.get_rank()
    
    # Each rank processes a different subset of data
    # Simulate splitting a dataset across DP ranks
    x = torch.randn(batch_size, 128, device=device)
    
    # Forward pass (no communication needed during inference)
    with torch.no_grad():
        output = model(x)
    
    print(f"Rank {dp_rank}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Each rank has a full model copy (no sharding)")
    
    # Verify that different ranks process different data
    # In real scenarios, outputs would be different because inputs are different
    if dp_size > 1:
        # Gather outputs from all ranks to show they're different
        output_list = [torch.zeros_like(output) for _ in range(dp_size)]
        dist.all_gather(output_list, output)
        
        # Check if outputs are different (they should be, since inputs are different)
        all_different = not all(torch.allclose(output_list[0], out, atol=1e-4) for out in output_list[1:])
        print(f"  Outputs are different across ranks: {all_different} (expected: True)")


def demo_data_parallel_training(device: torch.device, dp_size: int):
    """Demonstrate data parallel training with gradient synchronization"""
    print("\n" + "="*60)
    print("Demo 2: Data Parallel Training (Gradient Synchronization)")
    print("="*60)
    
    # Create a model
    model = SimpleModel(input_size=128, hidden_size=256, output_size=10).to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create different data for each rank
    batch_size = 4
    dp_rank = dist.get_rank()
    x = torch.randn(batch_size, 128, device=device)
    # Create dummy targets
    targets = torch.randint(0, 10, (batch_size,), device=device)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Synchronize gradients across all DP ranks
    # In data parallelism, we average gradients from all ranks
    for param in model.parameters():
        if param.grad is not None:
            if dp_size > 1:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dp_size  # Average the gradients
            # For single process, no synchronization needed (gradient is already correct)
    
    # Update parameters
    optimizer.step()
    
    print(f"Rank {dp_rank}:")
    print(f"  Loss: {loss.item():.4f}")
    if dp_size > 1:
        print(f"  Gradients synchronized across {dp_size} ranks")
    else:
        print(f"  Single process mode: no gradient synchronization needed")
    print(f"  Parameters updated with averaged gradients")


def demo_transformer_data_parallel(device: torch.device, dp_size: int):
    """Demonstrate data parallel inference with transformer model"""
    print("\n" + "="*60)
    print("Demo 3: Data Parallel Transformer Inference")
    print("="*60)
    
    # Create a transformer model (each rank has a full copy)
    model = SimpleTransformer(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
        max_seq_len=128
    ).to(device)
    model.eval()
    
    # Create different input sequences for each rank
    batch_size = 2
    seq_len = 10
    dp_rank = dist.get_rank()
    
    # Each rank processes different sequences
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
    
    # Get predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    
    print(f"Rank {dp_rank}:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Processing different sequences on each rank")


def demo_throughput_benefits(device: torch.device, dp_size: int):
    """Demonstrate throughput benefits of data parallelism"""
    print("\n" + "="*60)
    print("Demo 4: Throughput Benefits")
    print("="*60)
    
    model = SimpleModel(input_size=128, hidden_size=256, output_size=10).to(device)
    model.eval()
    
    # Simulate processing multiple batches
    num_batches = 10
    batch_size = 4
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            x = torch.randn(batch_size, 128, device=device)
            _ = model(x)
    
    elapsed_time = time.time() - start_time
    
    dp_rank = dist.get_rank()
    print(f"Rank {dp_rank}:")
    print(f"  Processed {num_batches} batches of size {batch_size}")
    print(f"  Time per batch: {elapsed_time / num_batches * 1000:.2f} ms")
    print(f"  Total throughput: {num_batches * batch_size / elapsed_time:.2f} samples/sec")
    print(f"  With DP={dp_size}, total system throughput: ~{num_batches * batch_size * dp_size / elapsed_time:.2f} samples/sec")


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallelism Demo")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPUs are available"
    )
    args = parser.parse_args()
    
    # Use shared mdaisy utility for distributed initialization
    rank, world_size, device, local_rank = init_distributed(use_cpu=args.force_cpu)
    
    # Determine if using CPU
    use_cpu = device.type == "cpu"
    
    # Initialize data parallelism
    dp_size = 2  # Use 2 processes for data parallelism
    if world_size >= dp_size:
        # Use appropriate backend based on device
        backend = "gloo" if use_cpu else "nccl"
        initialize_data_parallel(dp_size, backend=backend)
        actual_dp_size = dp_size
    else:
        print(f"Note: world_size ({world_size}) < dp_size ({dp_size}), using world_size for debugging")
        backend = "gloo" if use_cpu else "nccl"
        initialize_data_parallel(world_size, backend=backend)
        actual_dp_size = world_size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Data Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Data parallel size: {actual_dp_size}")
        print(f"Device: {device}")
        print(f"Backend: {'gloo (CPU)' if use_cpu else 'nccl (GPU)'}")
        if use_cpu and torch.cuda.is_available():
            print(f"Note: Using CPU mode (GPUs available: {torch.cuda.device_count()})")
    
    # Run demos (allow single process for debugging)
    if actual_dp_size > 1:
        demo_data_parallel_inference(device, actual_dp_size)
        demo_data_parallel_training(device, actual_dp_size)
        demo_transformer_data_parallel(device, actual_dp_size)
        demo_throughput_benefits(device, actual_dp_size)
    else:
        print("\nNote: Running in single-process mode (useful for debugging)")
        print("For full data parallelism demo, run with: torchrun --nproc_per_node=2 demo.py")
        print("\nRunning demos in single-process mode...")
        demo_data_parallel_inference(device, actual_dp_size)
        demo_data_parallel_training(device, actual_dp_size)
        demo_transformer_data_parallel(device, actual_dp_size)
        demo_throughput_benefits(device, actual_dp_size)
    
    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("\n" + "="*60)
            print("Demo completed!")
            print("="*60)
        
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

