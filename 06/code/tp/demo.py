"""
Tensor Parallelism Demo
Demonstrates column parallelism, row parallelism, and MLP with TP

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

from parallel_state import initialize_tensor_parallel
from linear import ColumnParallelLinear, RowParallelLinear
from mlp import TensorParallelMLP


def demo_column_parallel(device: torch.device, tp_size: int):
    """Demonstrate column parallel linear layer"""
    print("\n" + "="*60)
    print("Demo 1: Column Parallel Linear Layer")
    print("="*60)
    
    input_size = 128
    output_size = 256
    
    # Create column parallel linear layer
    layer = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        gather_output=False,  # Don't gather to see sharded output
    ).to(device)
    
    # Create input
    batch_size = 4
    x = torch.randn(batch_size, input_size, device=device)
    
    # Forward pass
    output = layer(x)
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape} (sharded, should be [{batch_size}, {output_size // tp_size}])")
    print(f"  Weight shape: {layer.weight.shape} (sharded)")
    
    # Now with gather_output=True
    layer_gather = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        gather_output=True,  # Gather to get full output
    ).to(device)
    layer_gather.weight.data.copy_(layer.weight.data)
    layer_gather.bias.data.copy_(layer.bias.data)
    
    output_gathered = layer_gather(x)
    print(f"  Output shape (gathered): {output_gathered.shape} (full, should be [{batch_size}, {output_size}])")


def demo_row_parallel(device: torch.device, tp_size: int):
    """Demonstrate row parallel linear layer"""
    print("\n" + "="*60)
    print("Demo 2: Row Parallel Linear Layer")
    print("="*60)
    
    input_size = 256
    output_size = 128
    
    # Create row parallel linear layer
    layer = RowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        input_is_parallel=True,  # Input is already sharded
        reduce_results=True,  # All-reduce to get final result
    ).to(device)
    
    # Create sharded input (simulating output from column parallel layer)
    batch_size = 4
    input_size_per_partition = input_size // tp_size
    x_sharded = torch.randn(batch_size, input_size_per_partition, device=device)
    
    # Forward pass
    output = layer(x_sharded)
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Input shape (sharded): {x_sharded.shape} (should be [{batch_size}, {input_size_per_partition}])")
    print(f"  Output shape: {output.shape} (full, should be [{batch_size}, {output_size}])")
    print(f"  Weight shape: {layer.weight.shape} (sharded)")


def demo_mlp(device: torch.device, tp_size: int):
    """Demonstrate tensor parallel MLP"""
    print("\n" + "="*60)
    print("Demo 3: Tensor Parallel MLP")
    print("="*60)
    
    hidden_size = 128
    intermediate_size = 512  # Typically 4x hidden_size
    
    # Create MLP
    mlp = TensorParallelMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    ).to(device)
    
    # Create input
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Forward pass
    output = mlp(x)
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape} (should be [{batch_size}, {seq_len}, {hidden_size}])")
    print(f"  Gate proj weight shape: {mlp.gate_proj.weight.shape} (column parallel shard)")
    print(f"  Down proj weight shape: {mlp.down_proj.weight.shape} (row parallel shard)")
    
    # Verify output is the same on all ranks (after all-reduce)
    if tp_size > 1:
        # Gather outputs from all ranks to verify they're the same
        output_list = [torch.zeros_like(output) for _ in range(tp_size)]
        dist.all_gather(output_list, output)
        
        # Check if all outputs are the same
        all_same = all(torch.allclose(output_list[0], out) for out in output_list[1:])
        print(f"  Outputs are same across ranks: {all_same}")


def demo_memory_benefits(device: torch.device, tp_size: int):
    """Demonstrate memory benefits of tensor parallelism"""
    print("\n" + "="*60)
    print("Demo 4: Memory Benefits")
    print("="*60)
    
    # Simulate a large model layer
    hidden_size = 4096
    intermediate_size = 16384  # 4x hidden_size
    
    mlp = TensorParallelMLP(hidden_size, intermediate_size).to(device)
    
    # Calculate parameter count
    gate_params = mlp.gate_proj.weight.numel() + (mlp.gate_proj.bias.numel() if mlp.gate_proj.bias is not None else 0)
    up_params = mlp.up_proj.weight.numel() + (mlp.up_proj.bias.numel() if mlp.up_proj.bias is not None else 0)
    down_params = mlp.down_proj.weight.numel() + (mlp.down_proj.bias.numel() if mlp.down_proj.bias is not None else 0)
    total_params = gate_params + up_params + down_params
    
    # Full model parameters (if no TP)
    full_gate_params = hidden_size * intermediate_size
    full_up_params = hidden_size * intermediate_size
    full_down_params = intermediate_size * hidden_size
    full_total_params = full_gate_params + full_up_params + full_down_params
    
    print(f"Rank {dist.get_rank()}:")
    print(f"  Parameters per GPU (with TP={tp_size}): {total_params:,}")
    print(f"  Parameters per GPU (without TP): {full_total_params:,}")
    print(f"  Memory reduction: {full_total_params / total_params:.2f}x")
    print(f"  Weight memory (FP16): {total_params * 2 / 1024**2:.2f} MB per GPU")


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="Tensor Parallelism Demo")
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
    
    # Initialize tensor parallelism
    tp_size = 2  # Use 2 processes for tensor parallelism
    if world_size >= tp_size:
        # Use appropriate backend based on device
        backend = "gloo" if use_cpu else "nccl"
        initialize_tensor_parallel(tp_size, backend=backend)
        actual_tp_size = tp_size
    else:
        print(f"Warning: world_size ({world_size}) < tp_size ({tp_size}), using world_size")
        backend = "gloo" if use_cpu else "nccl"
        initialize_tensor_parallel(world_size, backend=backend)
        actual_tp_size = world_size
    
    if rank == 0:
        print("\n" + "="*60)
        print("Tensor Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Tensor parallel size: {actual_tp_size}")
        print(f"Device: {device}")
        print(f"Backend: {'gloo (CPU)' if use_cpu else 'nccl (GPU)'}")
        if use_cpu and torch.cuda.is_available():
            print(f"Note: Using CPU mode (GPUs available: {torch.cuda.device_count()})")
    
    # Run demos
    if actual_tp_size > 1:
        demo_column_parallel(device, actual_tp_size)
        demo_row_parallel(device, actual_tp_size)
        demo_mlp(device, actual_tp_size)
        demo_memory_benefits(device, actual_tp_size)
    else:
        print("\nWarning: Need at least 2 processes for tensor parallelism demo")
        print("Run with: torchrun --nproc_per_node=2 demo.py")
    
    # Cleanup
    # Ensure we're on the correct device before barrier to avoid warning
    # mdaisy's init_distributed already handles device setup, but we ensure it's set here too
    if not use_cpu and torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Demo completed!")
        print("="*60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

