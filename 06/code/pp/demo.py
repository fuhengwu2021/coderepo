"""
Pipeline Parallelism Demo
Demonstrates pipeline parallelism where model layers are split across ranks

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

from parallel_state import (
    initialize_pipeline_parallel,
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    get_prev_rank,
    get_next_rank,
    is_first_stage,
    is_last_stage,
)
from pipeline_model import (
    PipelineStage,
    TransformerBlock,
    create_pipeline_stages,
    PipelineParallelModel,
)


def demo_pipeline_forward(device: torch.device, pp_size: int):
    """Demonstrate pipeline parallel forward pass"""
    print("\n" + "="*60)
    print("Demo 1: Pipeline Parallel Forward Pass")
    print("="*60)
    
    # Model configuration
    num_layers = 4
    hidden_size = 128
    num_heads = 4
    intermediate_size = 512
    
    # Create stages for this rank
    stage_idx = get_pipeline_parallel_rank()
    all_stages = create_pipeline_stages(
        num_layers, hidden_size, num_heads, intermediate_size, pp_size
    )
    
    # Create this rank's stage
    stage = PipelineStage(all_stages[stage_idx], stage_idx, pp_size)
    stage = stage.to(device)
    stage.eval()
    
    print(f"Rank {dist.get_rank()} (Stage {stage_idx}):")
    print(f"  Number of layers in this stage: {len(stage.layers)}")
    print(f"  Previous stage rank: {get_prev_rank()}")
    print(f"  Next stage rank: {get_next_rank()}")
    print(f"  Is first stage: {is_first_stage()}")
    print(f"  Is last stage: {is_last_stage()}")
    
    # Create input (only first stage needs input)
    batch_size = 2
    seq_len = 10
    
    if is_first_stage():
        # First stage: create input
        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        print(f"  Input shape: {x.shape}")
    else:
        # Other stages: receive input from previous stage
        prev_rank = get_prev_rank()
        if prev_rank is not None:
            x = torch.empty(batch_size, seq_len, hidden_size, device=device)
            dist.recv(x, src=prev_rank)
            print(f"  Received input shape: {x.shape}")
        else:
            raise RuntimeError("Non-first stage must have previous rank")
    
    # Forward through this stage
    with torch.no_grad():
        output = stage(x)
    
    # Send output to next stage (if not last stage)
    if not is_last_stage():
        next_rank = get_next_rank()
        if next_rank is not None:
            dist.send(output, dst=next_rank)
            print(f"  Sent output to stage {next_rank}")
    else:
        # Last stage: output is final result
        print(f"  Final output shape: {output.shape}")
        print(f"  Pipeline forward pass completed!")


def demo_pipeline_microbatches(device: torch.device, pp_size: int):
    """Demonstrate pipeline parallelism with microbatches"""
    print("\n" + "="*60)
    print("Demo 2: Pipeline Parallelism with Microbatches")
    print("="*60)
    
    # Model configuration
    num_layers = 6
    hidden_size = 128
    num_heads = 4
    intermediate_size = 512
    num_microbatches = 4
    
    # Create stages
    stage_idx = get_pipeline_parallel_rank()
    all_stages = create_pipeline_stages(
        num_layers, hidden_size, num_heads, intermediate_size, pp_size
    )
    
    stage = PipelineStage(all_stages[stage_idx], stage_idx, pp_size)
    stage = stage.to(device)
    stage.eval()
    
    batch_size = 2
    seq_len = 10
    
    print(f"Rank {dist.get_rank()} (Stage {stage_idx}):")
    print(f"  Processing {num_microbatches} microbatches")
    
    # Process microbatches in pipeline fashion
    with torch.no_grad():
        for microbatch_idx in range(num_microbatches):
            if is_first_stage():
                # First stage: create input for this microbatch
                x = torch.randn(batch_size, seq_len, hidden_size, device=device)
            else:
                # Receive from previous stage
                prev_rank = get_prev_rank()
                if prev_rank is not None:
                    x = torch.empty(batch_size, seq_len, hidden_size, device=device)
                    dist.recv(x, src=prev_rank)
            
            # Forward through this stage
            output = stage(x)
            
            # Send to next stage
            if not is_last_stage():
                next_rank = get_next_rank()
                if next_rank is not None:
                    dist.send(output, dst=next_rank)
            else:
                # Last stage: collect final outputs
                if microbatch_idx == 0:
                    print(f"  Microbatch {microbatch_idx}: output shape {output.shape}")
    
    dist.barrier()
    if is_last_stage():
        print(f"  All {num_microbatches} microbatches processed!")


def demo_memory_benefits(device: torch.device, pp_size: int):
    """Demonstrate memory benefits of pipeline parallelism"""
    print("\n" + "="*60)
    print("Demo 3: Memory Benefits of Pipeline Parallelism")
    print("="*60)
    
    # Model configuration
    num_layers = 8
    hidden_size = 512
    num_heads = 8
    intermediate_size = 2048
    
    stage_idx = get_pipeline_parallel_rank()
    all_stages = create_pipeline_stages(
        num_layers, hidden_size, num_heads, intermediate_size, pp_size
    )
    
    stage = PipelineStage(all_stages[stage_idx], stage_idx, pp_size)
    stage = stage.to(device)
    
    # Calculate parameters in this stage
    stage_params = sum(p.numel() for p in stage.parameters())
    
    # Estimate full model parameters (if no PP)
    layers_per_stage = num_layers // pp_size
    # Each transformer block has roughly:
    # - Attention: 4 * hidden_size^2 (q, k, v, o projections)
    # - MLP: 3 * hidden_size * intermediate_size (gate, up, down)
    # - Layer norms: 2 * hidden_size (negligible)
    params_per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    full_model_params = num_layers * params_per_layer
    
    print(f"Rank {dist.get_rank()} (Stage {stage_idx}):")
    print(f"  Parameters in this stage: {stage_params:,}")
    print(f"  Parameters in full model (no PP): {full_model_params:,}")
    print(f"  Memory reduction per GPU: {full_model_params / stage_params:.2f}x")
    print(f"  Weight memory (FP16) per GPU: {stage_params * 2 / 1024**2:.2f} MB")


def demo_pipeline_efficiency(device: torch.device, pp_size: int):
    """Demonstrate pipeline efficiency considerations"""
    print("\n" + "="*60)
    print("Demo 4: Pipeline Efficiency Considerations")
    print("="*60)
    
    stage_idx = get_pipeline_parallel_rank()
    
    print(f"Rank {dist.get_rank()} (Stage {stage_idx}):")
    print(f"  Pipeline stages: {pp_size}")
    print(f"  Pipeline bubbles: {pp_size - 1} (idle time during pipeline fill/drain)")
    print(f"  Efficiency improves with more microbatches")
    print(f"  Communication overhead: {pp_size - 1} activations transfers per forward pass")
    
    if is_first_stage():
        print(f"  First stage: processes input, no receive overhead")
    elif is_last_stage():
        print(f"  Last stage: produces output, no send overhead")
    else:
        print(f"  Intermediate stage: receives from prev, sends to next")


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Demo")
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
    
    # Initialize pipeline parallelism
    pp_size = 2  # Use 2 processes for pipeline parallelism
    if world_size >= pp_size:
        # Use appropriate backend based on device
        backend = "gloo" if use_cpu else "nccl"
        initialize_pipeline_parallel(pp_size, backend=backend)
        actual_pp_size = pp_size
    else:
        print(f"Warning: world_size ({world_size}) < pp_size ({pp_size}), using world_size")
        backend = "gloo" if use_cpu else "nccl"
        initialize_pipeline_parallel(world_size, backend=backend)
        actual_pp_size = world_size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Pipeline Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Pipeline parallel size: {actual_pp_size}")
        print(f"Device: {device}")
        print(f"Backend: {'gloo (CPU)' if use_cpu else 'nccl (GPU)'}")
        if use_cpu and torch.cuda.is_available():
            print(f"Note: Using CPU mode (GPUs available: {torch.cuda.device_count()})")
    
    # Run demos
    if actual_pp_size > 1:
        demo_pipeline_forward(device, actual_pp_size)
        dist.barrier()  # Synchronize between demos
        demo_pipeline_microbatches(device, actual_pp_size)
        dist.barrier()
        demo_memory_benefits(device, actual_pp_size)
        dist.barrier()
        demo_pipeline_efficiency(device, actual_pp_size)
    else:
        print("\nWarning: Need at least 2 processes for pipeline parallelism demo")
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

