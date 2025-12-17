"""
Demo script for TP RadixAttention.

This demonstrates how Tensor Parallelism integrates with RadixAttention
following SGLang's approach.

Usage:
    # Run with 2 TP ranks
    torchrun --nproc_per_node=2 demo_tp_radix.py
    
    # Run with 4 TP ranks
    torchrun --nproc_per_node=4 demo_tp_radix.py
    
    # Force CPU (if no GPUs available)
    torchrun --nproc_per_node=2 demo_tp_radix.py --force-cpu
"""
import torch
import torch.distributed as dist
import argparse
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tp.parallel_state import initialize_tensor_parallel
from tp.tp_model_wrapper import TPRadixAttentionModelWrapper
from tp.scheduler import TPScheduler


def main():
    parser = argparse.ArgumentParser(description="TP RadixAttention Demo")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force CPU usage even if GPUs are available")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size (default: world_size)")
    parser.add_argument("--page-size", type=int, default=16,
                        help="Page size for RadixCache")
    args = parser.parse_args()
    
    # Initialize distributed
    if not dist.is_initialized():
        # Determine backend
        use_cpu = args.force_cpu or not torch.cuda.is_available() or torch.cuda.device_count() < 2
        backend = "gloo" if use_cpu else "nccl"
        
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Running with torchrun
            dist.init_process_group(backend=backend)
        else:
            # Single process - initialize for testing
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group(backend=backend, rank=0, world_size=1)
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Determine TP size
    tp_size = args.tp_size if args.tp_size is not None else world_size
    
    if tp_size > world_size:
        if rank == 0:
            print(f"Warning: tp_size ({tp_size}) > world_size ({world_size}), using world_size")
        tp_size = world_size
    
    if world_size % tp_size != 0:
        if rank == 0:
            print(f"Error: world_size ({world_size}) must be divisible by tp_size ({tp_size})")
        sys.exit(1)
    
    # Initialize TP
    use_cpu = args.force_cpu or not torch.cuda.is_available()
    backend = "gloo" if use_cpu else "nccl"
    initialize_tensor_parallel(tensor_parallel_size=tp_size, backend=backend)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TP RadixAttention Demo")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"TP size: {tp_size}")
        print(f"Backend: {backend}")
        print(f"Device: {'CPU' if use_cpu else 'CUDA'}")
        print(f"{'='*60}\n")
    
    # Create TP RadixAttention model wrapper
    device = "cpu" if use_cpu else "cuda"
    model_wrapper = TPRadixAttentionModelWrapper(
        model_name=args.model_name,
        device=device,
        page_size=args.page_size,
    )
    
    # Create scheduler
    scheduler = TPScheduler(model_wrapper)
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "What is the capital of France?",  # Same prompt to test prefix sharing
        "Explain quantum computing in simple terms.",
    ]
    
    if rank == 0:
        print("\n" + "="*60)
        print("Testing TP RadixAttention")
        print("="*60)
    
    # Process requests
    for i, prompt in enumerate(test_prompts):
        request_id = scheduler.add_request(prompt, max_new_tokens=10)
        if rank == 0:
            print(f"\n[Test {i+1}] Added request {request_id}: {prompt}")
    
    # Process all requests
    results = scheduler.process_requests()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Results:")
        print("="*60)
        for i, result in enumerate(results):
            print(f"\n[Result {i+1}]:")
            print(f"  Prompt: {test_prompts[i]}")
            print(f"  Generated: {result}")
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)


if __name__ == "__main__":
    main()
