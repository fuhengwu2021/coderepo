"""
Data Parallelism Demo for vLLM-style Inference
Demonstrates data parallelism where each rank has a full model replica
and processes independent request streams concurrently

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

from parallel_state import (
    initialize_data_parallel,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)
from model import SimpleModel, SimpleTransformer


def demo_data_parallel_inference(device: torch.device, dp_size: int):
    """Demonstrate data parallel inference with request streams"""
    print("\n" + "="*60)
    print("Demo 1: Data Parallel Inference (Request Streams)")
    print("="*60)
    
    # Create a model (each rank has a full replica)
    model = SimpleModel(input_size=128, hidden_size=256, output_size=10).to(device)
    model.eval()
    
    # Use DP rank instead of global rank for clarity (matches vLLM semantics)
    dp_rank = get_data_parallel_rank()
    
    # Simulate independent request streams on each DP rank
    # In vLLM, each DP replica processes different requests concurrently
    num_requests = 4
    batch_size = 1  # Each request is processed independently
    
    print(f"Rank {dp_rank}: Processing {num_requests} independent requests")
    print(f"  Each DP rank represents an independent inference replica")
    print(f"  No communication needed during inference (vLLM-style)")
    
    # Process requests independently on this rank
    request_outputs = []
    with torch.no_grad():
        for request_id in range(num_requests):
            # Each rank processes different requests (simulated by different random inputs)
            # In real vLLM, these would come from a scheduler
            x = torch.randn(batch_size, 128, device=device)
            output = model(x)
            request_outputs.append(output)
            
            if request_id == 0:  # Show details for first request
                print(f"  Request {request_id}: input shape {x.shape}, output shape {output.shape}")
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Each rank has a full model replica (no sharding)")
    print(f"  Processed {num_requests} requests independently")
    
    # In vLLM, outputs from different ranks are independent
    # No need to gather/synchronize during inference




def demo_transformer_data_parallel(device: torch.device, dp_size: int):
    """Demonstrate data parallel inference with transformer model"""
    print("\n" + "="*60)
    print("Demo 2: Data Parallel Transformer Inference (Request Streams)")
    print("="*60)
    
    # Create a transformer model (each rank has a full replica)
    model = SimpleTransformer(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
        max_seq_len=128
    ).to(device)
    model.eval()
    
    # Use DP rank instead of global rank for clarity (matches vLLM semantics)
    dp_rank = get_data_parallel_rank()
    
    # Simulate processing different request streams on each rank
    # In vLLM, each DP replica handles independent request batches
    num_requests = 3
    seq_len = 10
    
    print(f"Rank {dp_rank}: Processing {num_requests} independent request streams")
    
    with torch.no_grad():
        for request_id in range(num_requests):
            # Each request has different input sequences
            # In vLLM, these would come from different users/clients
            input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
            logits = model(input_ids)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            if request_id == 0:  # Show details for first request
                print(f"  Request {request_id}: input shape {input_ids.shape}, logits shape {logits.shape}")
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Each rank processes independent request streams (no cross-rank communication)")


def demo_throughput_benefits(device: torch.device, dp_size: int):
    """Demonstrate throughput benefits of data parallelism"""
    print("\n" + "="*60)
    print("Demo 3: Throughput Benefits of Data Parallelism")
    print("="*60)
    
    model = SimpleModel(input_size=128, hidden_size=256, output_size=10).to(device)
    model.eval()
    
    # Simulate processing multiple requests concurrently
    num_requests = 20
    requests_per_batch = 1  # Each request processed independently
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_requests):
            # Each rank processes independent requests
            x = torch.randn(requests_per_batch, 128, device=device)
            _ = model(x)
    
    elapsed_time = time.time() - start_time
    
    # Use DP rank instead of global rank for clarity (matches vLLM semantics)
    dp_rank = get_data_parallel_rank()
    print(f"Rank {dp_rank}:")
    print(f"  Processed {num_requests} independent requests")
    print(f"  Time per request: {elapsed_time / num_requests * 1000:.2f} ms")
    print(f"  Throughput on this rank: {num_requests / elapsed_time:.2f} requests/sec")
    print(f"  With DP={dp_size}, system throughput scales: ~{num_requests * dp_size / elapsed_time:.2f} requests/sec")
    print(f"  Note: Each DP replica processes independent request streams concurrently")


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
    # Note: DP group exists for uniformity with TP/PP; inference does not require collectives.
    # In vLLM-style DP, each replica processes independent requests with no cross-rank communication.
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
        demo_transformer_data_parallel(device, actual_dp_size)
        demo_throughput_benefits(device, actual_dp_size)
    else:
        print("\nNote: Running in single-process mode (useful for debugging)")
        print("For full data parallelism demo, run with: torchrun --nproc_per_node=2 demo.py")
        print("\nRunning demos in single-process mode...")
        demo_data_parallel_inference(device, actual_dp_size)
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

