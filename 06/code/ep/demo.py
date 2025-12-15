"""
Expert Parallelism Demo
Demonstrates expert parallelism for Mixture-of-Experts (MoE) models

Run with:
    torchrun --nproc_per_node=2 demo.py
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

from parallel_state import initialize_expert_parallel
from moe import ExpertParallelMoE, create_moe


def demo_expert_parallelism(device: torch.device, ep_size: int):
    """Demonstrate expert parallel MoE layer"""
    print("\n" + "="*60)
    print("Demo: Expert Parallelism for MoE")
    print("="*60)
    
    num_experts = 8
    hidden_size = 128
    intermediate_size = 512
    top_k = 2
    
    # Create MoE layer
    moe = ExpertParallelMoE(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
    ).to(device)
    
    # Create input
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size * seq_len, hidden_size, device=device)
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Total experts: {num_experts}")
    print(f"  Local experts: {moe.num_local_experts} (experts [{moe.expert_start_idx}, {moe.expert_end_idx}))")
    print(f"  EP size: {ep_size}")
    
    # Forward pass
    output = moe(x)
    
    print(f"  Output shape: {output.shape} (should be [{batch_size * seq_len}, {hidden_size}])")
    
    # Verify output is the same on all ranks (after reduce-scatter)
    if ep_size > 1:
        # Gather outputs from all ranks to verify they're the same
        output_list = [torch.zeros_like(output) for _ in range(ep_size)]
        dist.all_gather(output_list, output)
        
        # Check if all outputs are the same
        all_same = all(torch.allclose(output_list[0], out, atol=1e-4) for out in output_list[1:])
        print(f"  Outputs are same across ranks: {all_same}")


def demo_memory_benefits(device: torch.device, ep_size: int):
    """Demonstrate memory benefits of expert parallelism"""
    print("\n" + "="*60)
    print("Demo: Memory Benefits of Expert Parallelism")
    print("="*60)
    
    num_experts = 8
    hidden_size = 4096
    intermediate_size = 16384
    
    moe = ExpertParallelMoE(num_experts, hidden_size, intermediate_size).to(device)
    
    # Calculate parameter count
    router_params = sum(p.numel() for p in moe.router.parameters())
    expert_params = sum(p.numel() for expert in moe.experts for p in expert.parameters())
    total_params = router_params + expert_params
    
    # Full model parameters (if no EP)
    full_router_params = hidden_size * num_experts
    full_expert_params = num_experts * (
        hidden_size * intermediate_size +  # gate_proj
        hidden_size * intermediate_size +  # up_proj
        intermediate_size * hidden_size     # down_proj
    )
    full_total_params = full_router_params + full_expert_params
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Parameters per GPU (with EP={ep_size}): {total_params:,}")
    print(f"  Parameters per GPU (without EP): {full_total_params:,}")
    print(f"  Memory reduction: {full_total_params / total_params:.2f}x")
    print(f"  Weight memory (FP16): {total_params * 2 / 1024**2:.2f} MB per GPU")


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="Expert Parallelism Demo")
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
    
    # Initialize expert parallelism
    ep_size = 2  # Use 2 processes for expert parallelism
    if world_size >= ep_size:
        # Use appropriate backend based on device
        backend = "gloo" if use_cpu else "nccl"
        initialize_expert_parallel(ep_size, backend=backend)
        actual_ep_size = ep_size
    else:
        print(f"Warning: world_size ({world_size}) < ep_size ({ep_size}), using world_size")
        backend = "gloo" if use_cpu else "nccl"
        initialize_expert_parallel(world_size, backend=backend)
        actual_ep_size = world_size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Expert Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Expert parallel size: {actual_ep_size}")
        print(f"Device: {device}")
        print(f"Backend: {'gloo (CPU)' if use_cpu else 'nccl (GPU)'}")
        if use_cpu and torch.cuda.is_available():
            print(f"Note: Using CPU mode (GPUs available: {torch.cuda.device_count()})")
    
    # Run demos
    if actual_ep_size > 1:
        demo_expert_parallelism(device, actual_ep_size)
        demo_memory_benefits(device, actual_ep_size)
    else:
        print("\nWarning: Need at least 2 processes for expert parallelism demo")
        print("Run with: torchrun --nproc_per_node=2 demo.py")
    
    # Cleanup
    dist.barrier()
    if rank == 0:
        print("\n" + "="*60)
        print("Demo completed!")
        print("="*60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

