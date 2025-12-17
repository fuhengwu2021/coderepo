"""
vLLM-style TP Demo with Real Weight Loading

This demo shows how vLLM loads and shards real model weights for TP,
demonstrating the key difference from generic TP implementations.

Run with:
    torchrun --nproc_per_node=2 demo_vllm.py
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
from linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from mlp import TensorParallelMLP
from weight_loader import load_and_shard_state_dict, load_weight_for_layer
from model_wrapper import TPModelWrapper


def demo_weight_loading(device: torch.device, tp_size: int):
    """Demonstrate vLLM-style weight loading with sharding"""
    print("\n" + "="*60)
    print("Demo: vLLM-style Weight Loading")
    print("="*60)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    if dist.get_rank() == 0:
        print(f"Loading weights from {model_name}...")
        print("Key difference from generic TP:")
        print("  - Weights are SHARDED during loading (not after)")
        print("  - Each TP rank only stores its shard in memory")
        print("  - Matches vLLM's production approach")
    
    # Create TP model wrapper (loads and shards weights)
    wrapper = TPModelWrapper(
        model_name=model_name,
        device=str(device),
        dtype=torch.float16,
    )
    
    if dist.get_rank() == 0:
        print("\nWeight Loading Statistics:")
        print(f"  Total weights in checkpoint: {len(wrapper.state_dict)}")
        print(f"  TP size: {wrapper.tp_size}")
        print(f"  Each rank stores: 1/{wrapper.tp_size} of weights")
    
    # Demonstrate loading a specific layer
    if dist.get_rank() == 0:
        print("\n" + "-"*60)
        print("Example: Loading QKV projection for layer 0")
        print("-"*60)
    
    # Create QKV layer
    qkv_layer = QKVParallelLinear(
        hidden_size=wrapper.hidden_size,
        head_size=wrapper.head_dim,
        total_num_heads=wrapper.num_heads,
        total_num_kv_heads=wrapper.num_kv_heads,
        bias=False,
        params_dtype=wrapper.dtype,
    ).to(device)
    
    # Load weight from state dict
    # Try separate Q, K, V first (Qwen uses separate projections)
    q_key = "model.layers.0.self_attn.q_proj.weight"
    k_key = "model.layers.0.self_attn.k_proj.weight"
    v_key = "model.layers.0.self_attn.v_proj.weight"
    
    if all(k in wrapper.state_dict for k in [q_key, k_key, v_key]):
        if dist.get_rank() == 0:
            print(f"  Found separate Q/K/V projections (Qwen-style)")
            print(f"  Q weight shape: {wrapper.state_dict[q_key].shape}")
            print(f"  K weight shape: {wrapper.state_dict[k_key].shape}")
            print(f"  V weight shape: {wrapper.state_dict[v_key].shape}")
        qkv_layer.weight_loader_qkv(wrapper.state_dict[q_key], "q")
        qkv_layer.weight_loader_qkv(wrapper.state_dict[k_key], "k")
        qkv_layer.weight_loader_qkv(wrapper.state_dict[v_key], "v")
    else:
        # Try fused QKV
        qkv_key = "model.layers.0.self_attn.qkv_proj.weight"
        if qkv_key in wrapper.state_dict:
            if dist.get_rank() == 0:
                print(f"  Found fused QKV projection")
            full_weight = wrapper.state_dict[qkv_key]
            if dist.get_rank() == 0:
                print(f"  Full weight shape: {full_weight.shape}")
            qkv_layer.weight_loader_qkv(full_weight, loaded_shard_id=None)
        else:
            if dist.get_rank() == 0:
                print(f"  Warning: Could not find QKV weights")
    
    if dist.get_rank() == 0:
        print(f"  Sharded weight shape on rank 0: {qkv_layer.weight.shape}")
        print(f"  Memory per rank: {qkv_layer.weight.numel() * 2 / 1024**2:.2f} MB (FP16)")
    
    # Demonstrate MLP weight loading
    if dist.get_rank() == 0:
        print("\n" + "-"*60)
        print("Example: Loading MLP layer 0")
        print("-"*60)
    
    mlp = TensorParallelMLP(
        hidden_size=wrapper.hidden_size,
        intermediate_size=wrapper.intermediate_size,
    ).to(device)
    
    # Load MLP weights
    prefix = "model.layers.0.mlp"
    gate_key = f"{prefix}.gate_proj.weight"
    up_key = f"{prefix}.up_proj.weight"
    down_key = f"{prefix}.down_proj.weight"
    
    if gate_key in wrapper.state_dict:
        mlp.gate_proj.weight_loader(wrapper.state_dict[gate_key])
        if dist.get_rank() == 0:
            print(f"  Loaded gate_proj: {wrapper.state_dict[gate_key].shape} -> {mlp.gate_proj.weight.shape}")
    
    if up_key in wrapper.state_dict:
        mlp.up_proj.weight_loader(wrapper.state_dict[up_key])
        if dist.get_rank() == 0:
            print(f"  Loaded up_proj: {wrapper.state_dict[up_key].shape} -> {mlp.up_proj.weight.shape}")
    
    if down_key in wrapper.state_dict:
        mlp.down_proj.weight_loader(wrapper.state_dict[down_key])
        if dist.get_rank() == 0:
            print(f"  Loaded down_proj: {wrapper.state_dict[down_key].shape} -> {mlp.down_proj.weight.shape}")
    
    # Show memory benefits
    if dist.get_rank() == 0:
        print("\n" + "-"*60)
        print("Memory Benefits (vLLM-style TP):")
        print("-"*60)
        stats = wrapper.get_memory_usage()
        print(f"  Parameters per rank: {stats['total_parameters']:,}")
        print(f"  Memory per rank: {stats['memory_per_rank_mb']:.2f} MB")
        print(f"  Total memory (all ranks): {stats['memory_per_rank_mb'] * tp_size:.2f} MB")
        print(f"  Memory reduction vs single GPU: {tp_size}x less per rank")


def demo_forward_pass(device: torch.device, tp_size: int):
    """Demonstrate forward pass with loaded weights"""
    print("\n" + "="*60)
    print("Demo: Forward Pass with Real Weights")
    print("="*60)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    wrapper = TPModelWrapper(model_name=model_name, device=str(device), dtype=torch.float16)
    
    # Create a simple test: one attention layer + one MLP layer
    try:
        from attention import TensorParallelAttention
    except ImportError:
        from .attention import TensorParallelAttention
    
    attn = wrapper.create_tp_attention_layer(0)
    mlp = wrapper.create_tp_mlp_layer(0)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    # Ensure input dtype matches layer dtype
    hidden_states = torch.randn(batch_size, seq_len, wrapper.hidden_size, device=device, dtype=wrapper.dtype)
    
    if dist.get_rank() == 0:
        print(f"Input shape: {hidden_states.shape}")
    
    # Forward through attention (inference_mode at top level)
    with torch.inference_mode():
        attn_output, kv_cache = attn(hidden_states)
        if dist.get_rank() == 0:
            print(f"Attention output shape: {attn_output.shape}")
            print(f"KV cache shapes: k={kv_cache[0].shape}, v={kv_cache[1].shape}")
        
        # Forward through MLP
        mlp_output = mlp(attn_output)
        if dist.get_rank() == 0:
            print(f"MLP output shape: {mlp_output.shape}")
            print("Forward pass completed successfully!")


def demo_text_generation(device: torch.device, tp_size: int):
    """Demonstrate actual text generation with Hybrid TP model"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Demo: Text Generation with Hybrid TP Model")
        print("="*60)
        print("Using hybrid approach:")
        print("  - TP layers: Attention and MLP (our implementation)")
        print("  - HF layers: Embedding, LayerNorm, LM Head")
        print("="*60)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    if rank == 0:
        print(f"\nLoading model {model_name}...")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create hybrid TP model (all ranks create it, but only rank 0 will use for generation)
    try:
        from hybrid_tp_model import HybridTPModel
    except ImportError:
        from .hybrid_tp_model import HybridTPModel
    
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    if rank == 0:
        print("Creating HybridTPModel (this may take a moment)...")
    
    # All ranks create the model (TP layers need to be on all ranks)
    model = HybridTPModel(
        model_name=model_name,
        device=str(device),
        dtype=dtype,
    )
    
    # Use the same prompts as inference_v4.py
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about AI.",
    ]
    
    if rank == 0:
        print(f"\nGenerating responses for {len(prompts)} prompts...")
        print("="*60)
    
    # Generate text for each prompt
    # IMPORTANT: All ranks must call generate() because TP requires all ranks to participate
    # Only rank 0 will print the results
    results = []
    
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            if rank == 0:
                print(f"\nPrompt {i+1}: {prompt}")
                print("-" * 60)
            
            # All ranks must call generate (TP requires all ranks to participate in forward)
            generated_text = model.generate(
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=50,
                do_sample=False,  # Greedy decoding for consistency
                temperature=1.0,
            )
            
            # Only rank 0 processes and prints results
            if rank == 0:
                # Extract response (remove prompt)
                response = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
                
                print(f"Response: {response}")
                results.append((prompt, response))
            
            # Synchronize after each prompt
            dist.barrier()
    
    # Final synchronization
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Generation Results Summary:")
        print("="*60)
        for i, (prompt, response) in enumerate(results):
            print(f"\n{i+1}. Prompt: {prompt}")
            print(f"   Response: {response}")
        
        print("\n" + "="*60)
        print("✓ Successfully generated text using Hybrid TP Model!")
        print("  - Attention and MLP layers use TP (sharded across ranks)")
        print("  - Other layers use HuggingFace original")
        print("="*60)


def main():
    """Main demo function"""
    import argparse
    parser = argparse.ArgumentParser(description="vLLM-style TP Demo with Real Weight Loading")
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if GPUs are available"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name"
    )
    args = parser.parse_args()
    
    # Use shared mdaisy utility for distributed initialization
    rank, world_size, device, local_rank = init_distributed(use_cpu=args.force_cpu)
    
    # Determine if using CPU
    use_cpu = device.type == "cpu"
    
    # Initialize tensor parallelism
    tp_size = 2
    if world_size >= tp_size:
        backend = "gloo" if use_cpu else "nccl"
        initialize_tensor_parallel(tp_size, backend=backend)
        actual_tp_size = tp_size
    else:
        if rank == 0:
            print(f"Warning: world_size ({world_size}) < tp_size ({tp_size}), using world_size")
        backend = "gloo" if use_cpu else "nccl"
        initialize_tensor_parallel(world_size, backend=backend)
        actual_tp_size = world_size
    
    if rank == 0:
        print("\n" + "="*60)
        print("vLLM-style Tensor Parallelism Demo")
        print("="*60)
        print(f"World size: {world_size}")
        print(f"Tensor parallel size: {actual_tp_size}")
        print(f"Device: {device}")
        print(f"Model: {args.model_name}")
        print("="*60)
        print("\nKey Features:")
        print("  ✓ Real weight loading from HuggingFace checkpoint")
        print("  ✓ Weights sharded during loading (vLLM-style)")
        print("  ✓ QKVParallelLinear for attention layers")
        print("  ✓ Proper weight_loader methods")
        print("  ✓ Inference-optimized (no autograd)")
    
    if actual_tp_size > 1:
        demo_weight_loading(device, actual_tp_size)
        demo_forward_pass(device, actual_tp_size)
        demo_text_generation(device, actual_tp_size)
    else:
        if rank == 0:
            print("\nWarning: Need at least 2 processes for tensor parallelism demo")
            print("Run with: torchrun --nproc_per_node=2 demo_vllm.py")
    
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("Demo completed!")
        print("="*60)
        print("\nComparison with Generic TP:")
        print("  Generic TP: Random initialization, then shard")
        print("  vLLM-style TP: Load from checkpoint, shard during loading")
        print("  → Matches production vLLM behavior")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

