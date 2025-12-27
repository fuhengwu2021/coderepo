"""
Compare RoPE application step by step
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from parallel_state import initialize_tensor_parallel

def test_rope_comparison():
    """Compare RoPE application in detail"""
    print("="*70)
    print("RoPE Application Comparison")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12360'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=str(device),
        dtype=dtype,
    )
    hf_model.eval()
    
    # Create test input
    batch_size = 1
    seq_len = 3
    hidden_states = torch.randn(batch_size, seq_len, 896, dtype=dtype).to(device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    
    # Get layer 0
    hf_layer0 = hf_model.model.layers[0]
    hf_attn = hf_layer0.self_attn
    
    # Apply layernorm
    hidden_ln = hf_layer0.input_layernorm(hidden_states)
    
    # QKV projections
    q = hf_attn.q_proj(hidden_ln)  # [1, 3, 896]
    k = hf_attn.k_proj(hidden_ln)  # [1, 3, 128]
    v = hf_attn.v_proj(hidden_ln)  # [1, 3, 128]
    
    print(f"\nAfter QKV proj: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # Reshape
    input_shape = hidden_ln.shape[:-1]
    q_reshaped = q.view(*input_shape, -1, 64).transpose(1, 2)  # [1, 14, 3, 64]
    k_reshaped = k.view(*input_shape, 2, 64).transpose(1, 2)  # [1, 2, 3, 64]
    v_reshaped = v.view(*input_shape, 2, 64).transpose(1, 2)  # [1, 2, 3, 64]
    
    print(f"After reshape: q={q_reshaped.shape}, k={k_reshaped.shape}, v={v_reshaped.shape}")
    
    # Get cos, sin
    rotary_emb = hf_model.model.rotary_emb
    k_for_rope = k_reshaped.transpose(1, 2)  # [1, 3, 2, 64]
    cos, sin = rotary_emb(k_for_rope, position_ids)
    print(f"cos: {cos.shape}, sin: {sin.shape}")
    
    # Apply RoPE
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    q_rope, k_rope = apply_rotary_pos_emb(q_reshaped, k_reshaped, cos, sin)
    print(f"After RoPE: q={q_rope.shape}, k={k_rope.shape}")
    
    # Now simulate what our TP attention does
    print("\n=== Simulating TP Attention ===")
    
    # Our TP attention does:
    # 1. QKV proj (fused) -> split
    # For tp=1, this should give same q, k, v
    # But we use QKVParallelLinear which concatenates Q/K/V
    
    # Let's manually create the fused QKV
    qkv_fused = torch.cat([q, k, v], dim=-1)  # [1, 3, 896+128+128] = [1, 3, 1152]
    print(f"Fused QKV: {qkv_fused.shape}")
    
    # Split
    q_from_fused = qkv_fused[:, :, :896]  # [1, 3, 896]
    k_from_fused = qkv_fused[:, :, 896:896+128]  # [1, 3, 128]
    v_from_fused = qkv_fused[:, :, 896+128:]  # [1, 3, 128]
    
    # Check if they match
    q_diff = (q - q_from_fused).abs().max().item()
    k_diff = (k - k_from_fused).abs().max().item()
    v_diff = (v - v_from_fused).abs().max().item()
    print(f"Q split diff: {q_diff:.6f}", "✅" if q_diff < 1e-5 else "❌")
    print(f"K split diff: {k_diff:.6f}", "✅" if k_diff < 1e-5 else "❌")
    print(f"V split diff: {v_diff:.6f}", "✅" if v_diff < 1e-5 else "❌")
    
    # Reshape (same as before)
    q_tp = q_from_fused.view(*input_shape, -1, 64).transpose(1, 2)
    k_tp = k_from_fused.view(*input_shape, 2, 64).transpose(1, 2)
    v_tp = v_from_fused.view(*input_shape, 2, 64).transpose(1, 2)
    
    # Apply RoPE (same way)
    k_for_rope_tp = k_tp.transpose(1, 2)
    cos_tp, sin_tp = rotary_emb(k_for_rope_tp, position_ids)
    q_rope_tp, k_rope_tp = apply_rotary_pos_emb(q_tp, k_tp, cos_tp, sin_tp)
    
    # Compare RoPE outputs
    q_rope_diff = (q_rope - q_rope_tp).abs().max().item()
    k_rope_diff = (k_rope - k_rope_tp).abs().max().item()
    print(f"\nRoPE output comparison:")
    print(f"  Q after RoPE diff: {q_rope_diff:.6f}", "✅" if q_rope_diff < 1e-5 else "❌")
    print(f"  K after RoPE diff: {k_rope_diff:.6f}", "✅" if k_rope_diff < 1e-5 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_rope_comparison()
