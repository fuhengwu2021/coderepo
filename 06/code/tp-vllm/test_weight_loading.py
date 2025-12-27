"""
Test weight loading: Verify that TP weights match HF weights after all_gather
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel, tensor_model_parallel_all_gather

def test_weight_loading():
    """Verify weight loading is correct"""
    print("="*70)
    print("Weight Loading Verification (tp=1)")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize distributed
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=str(device),
        dtype=dtype,
    )
    
    # Create TP model
    tp_model = HybridTPModel(
        model_name=model_name,
        device=str(device),
        dtype=dtype,
    )
    
    # Compare weights for first layer
    print("\nComparing Layer 0 Weights:")
    print("="*70)
    
    hf_layer0 = hf_model.model.layers[0]
    tp_layer0 = tp_model.layers[0]
    
    # Compare Q projection weights
    hf_q_weight = hf_layer0.self_attn.q_proj.weight.data  # [896, 896]
    tp_qkv_weight = tp_layer0.self_attn.qkv_proj.weight.data  # [1152, 896] for tp=1
    
    # For tp=1, QKV is [q_size + kv_size + kv_size, hidden_size] = [896 + 128 + 128, 896] = [1152, 896]
    # Q should be first 896 elements (14 heads * 64)
    tp_q_extracted = tp_qkv_weight[:896, :]  # Extract Q part
    
    # Compare
    max_diff = (hf_q_weight - tp_q_extracted).abs().max().item()
    mean_diff = (hf_q_weight - tp_q_extracted).abs().mean().item()
    
    print(f"Q projection:")
    print(f"  HF shape: {hf_q_weight.shape}")
    print(f"  TP shape (extracted): {tp_q_extracted.shape}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("  ✅ Q weights match!")
    else:
        print("  ❌ Q weights DO NOT match!")
        # Show some sample differences
        diff = (hf_q_weight - tp_q_extracted).abs()
        max_idx = diff.argmax()
        print(f"  Max diff at index {max_idx}: HF={hf_q_weight.flatten()[max_idx]:.6f}, TP={tp_q_extracted.flatten()[max_idx]:.6f}")
    
    # Compare K projection
    hf_k_weight = hf_layer0.self_attn.k_proj.weight.data  # [128, 896] (2 heads * 64)
    tp_k_extracted = tp_qkv_weight[896:896+128, :]  # Extract K part (2 heads * 64 for tp=1)
    
    print(f"\nK projection:")
    print(f"  HF shape: {hf_k_weight.shape}")
    print(f"  TP shape (extracted): {tp_k_extracted.shape}")
    
    max_diff_k = (hf_k_weight - tp_k_extracted).abs().max().item()
    mean_diff_k = (hf_k_weight - tp_k_extracted).abs().mean().item()
    print(f"  Max diff: {max_diff_k:.6f}")
    print(f"  Mean diff: {mean_diff_k:.6f}")
    
    if max_diff_k < 1e-3:
        print("  ✅ K weights match!")
    else:
        print("  ❌ K weights DO NOT match!")
    
    # Compare V projection
    hf_v_weight = hf_layer0.self_attn.v_proj.weight.data  # [128, 896]
    tp_v_extracted = tp_qkv_weight[896+128:896+128+128, :]  # Extract V part
    
    print(f"\nV projection:")
    print(f"  HF shape: {hf_v_weight.shape}")
    print(f"  TP shape (extracted): {tp_v_extracted.shape}")
    
    max_diff_v = (hf_v_weight - tp_v_extracted).abs().max().item()
    mean_diff_v = (hf_v_weight - tp_v_extracted).abs().mean().item()
    print(f"  Max diff: {max_diff_v:.6f}")
    print(f"  Mean diff: {mean_diff_v:.6f}")
    
    if max_diff_v < 1e-3:
        print("  ✅ V weights match!")
    else:
        print("  ❌ V weights DO NOT match!")
    
    # Compare MLP weights
    print(f"\nMLP Weights (Layer 0):")
    print("="*70)
    
    hf_mlp = hf_layer0.mlp
    tp_mlp = tp_layer0.mlp
    
    # Gate projection
    hf_gate = hf_mlp.gate_proj.weight.data
    tp_gate = tp_mlp.gate_proj.weight.data
    max_diff_gate = (hf_gate - tp_gate).abs().max().item()
    print(f"Gate projection: Max diff={max_diff_gate:.6f}", end="")
    print(" ✅" if max_diff_gate < 1e-3 else " ❌")
    
    # Up projection
    hf_up = hf_mlp.up_proj.weight.data
    tp_up = tp_mlp.up_proj.weight.data
    max_diff_up = (hf_up - tp_up).abs().max().item()
    print(f"Up projection: Max diff={max_diff_up:.6f}", end="")
    print(" ✅" if max_diff_up < 1e-3 else " ❌")
    
    # Down projection (RowParallel, should be full)
    hf_down = hf_mlp.down_proj.weight.data
    tp_down = tp_mlp.down_proj.weight.data
    max_diff_down = (hf_down - tp_down).abs().max().item()
    print(f"Down projection: Max diff={max_diff_down:.6f}", end="")
    print(" ✅" if max_diff_down < 1e-3 else " ❌")
    
    # Output projection (o_proj)
    hf_o = hf_layer0.self_attn.o_proj.weight.data
    tp_o = tp_layer0.self_attn.o_proj.weight.data
    max_diff_o = (hf_o - tp_o).abs().max().item()
    print(f"Output projection (o_proj): Max diff={max_diff_o:.6f}", end="")
    print(" ✅" if max_diff_o < 1e-3 else " ❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_weight_loading()
