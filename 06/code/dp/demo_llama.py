"""
Llama Data Parallelism Demo
Demonstrates data parallel inference with real Llama-3.2-1B-Instruct model

This demo shows how each DP replica loads a full model copy and processes
independent request streams, similar to vLLM's data parallelism.

Run with:
    torchrun --nproc_per_node=2 demo_llama.py
"""
import os
import sys
import torch
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

# For Llama demo
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def demo_llama_data_parallel(device: torch.device, dp_size: int):
    """Demonstrate data parallel inference with real Llama model"""
    if not TRANSFORMERS_AVAILABLE:
        print("\n" + "="*60)
        print("Demo 4: Llama-3.2-1B-Instruct Data Parallel Inference")
        print("="*60)
        print("  Skipped: transformers library not available")
        print("  Install with: pip install transformers accelerate")
        return
    
    print("\n" + "="*60)
    print("Demo 4: Llama-3.2-1B-Instruct Data Parallel Inference")
    print("="*60)
    
    # Use DP rank instead of global rank for clarity (matches vLLM semantics)
    dp_rank = get_data_parallel_rank()
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    if dp_rank == 0:
        print(f"Loading {model_name} on each DP replica...")
        print("  Note: Each rank loads a full model replica (no sharding)")
    
    # Load model and tokenizer on each rank
    # In vLLM, each DP replica has its own model instance
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate dtype and device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        model.eval()
        
        if dp_rank == 0:
            print(f"  Model loaded successfully")
            print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        if dp_rank == 0:
            print(f"  Error loading model: {e}")
            print(f"  Make sure you have:")
            print(f"    1. Access to the model (may need HuggingFace authentication)")
            print(f"    2. Sufficient GPU memory (model requires ~2GB per replica)")
            print(f"    3. Install: pip install transformers accelerate")
        return
    
    # Simulate different request streams on each DP rank
    # In vLLM, each replica processes independent requests
    requests = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI.",
        "What are the benefits of renewable energy?",
    ]
    
    # Each rank processes different requests
    # In real vLLM, requests are distributed by a scheduler
    rank_requests = requests[dp_rank::dp_size] if dp_size > 1 else requests
    
    print(f"\nRank {dp_rank}: Processing {len(rank_requests)} independent requests")
    print(f"  Each DP replica handles different request streams concurrently")
    
    import time
    total_tokens = 0
    
    with torch.no_grad():
        for i, prompt in enumerate(rank_requests):
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            # Generate (simple greedy decoding for demo)
            max_new_tokens = 50
            start_time = time.time()
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            elapsed = time.time() - start_time
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            num_tokens = outputs.shape[1] - input_ids.shape[1]
            total_tokens += num_tokens
            
            if i == 0:  # Show details for first request
                print(f"\n  Request {i}:")
                print(f"    Prompt: {prompt[:50]}...")
                print(f"    Generated: {generated_text[len(prompt):][:100]}...")
                print(f"    Tokens generated: {num_tokens}")
                print(f"    Time: {elapsed:.2f}s, Tokens/sec: {num_tokens/elapsed:.1f}")
    
    print(f"\n  Rank {dp_rank} summary:")
    print(f"    Total requests processed: {len(rank_requests)}")
    print(f"    Total tokens generated: {total_tokens}")
    print(f"    No cross-rank communication needed (vLLM-style DP)")
    
    if dp_size > 1:
        # Show that different ranks process different requests
        dist.barrier()
        if dp_rank == 0:
            print(f"\n  System-wide: {dp_size} replicas processing requests concurrently")
            print(f"    Throughput scales with number of DP replicas")


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="Llama Data Parallelism Demo")
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
        print("Llama Data Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Data parallel size: {actual_dp_size}")
        print(f"Device: {device}")
        print(f"Backend: {'gloo (CPU)' if use_cpu else 'nccl (GPU)'}")
        if use_cpu and torch.cuda.is_available():
            print(f"Note: Using CPU mode (GPUs available: {torch.cuda.device_count()})")
    
    # Run demo
    demo_llama_data_parallel(device, actual_dp_size)
    
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
