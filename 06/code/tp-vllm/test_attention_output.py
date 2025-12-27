"""
Test attention output: Compare TP attention with HF attention output
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def test_attention_output():
    """Compare attention output between HF and TP"""
    print("="*70)
    print("Attention Output Comparison")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load models
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=str(device),
        dtype=dtype,
    )
    hf_model.eval()
    
    tp_model = HybridTPModel(
        model_name=model_name,
        device=str(device),
        dtype=dtype,
    )
    tp_model.eval()
    
    # Create same input (ensure dtype matches)
    batch_size = 1
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, 896, dtype=dtype).to(device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    
    print(f"\nInput: hidden_states={hidden_states.shape}, position_ids={position_ids.shape}")
    
    # Get layer 0
    hf_layer0 = hf_model.model.layers[0]
    tp_layer0 = tp_model.layers[0]
    
    # Apply input_layernorm (should be same)
    hf_hidden_ln = hf_layer0.input_layernorm(hidden_states)
    tp_hidden_ln = tp_layer0.input_layernorm(hidden_states)
    
    max_diff_ln = (hf_hidden_ln - tp_hidden_ln).abs().max().item()
    print(f"\nAfter input_layernorm: Max diff={max_diff_ln:.6f}", end="")
    print(" ✅" if max_diff_ln < 1e-5 else " ❌")
    
    # Now manually do attention for both
    # HF attention
    hf_attn = hf_layer0.self_attn
    hf_q = hf_attn.q_proj(hf_hidden_ln)
    hf_k = hf_attn.k_proj(hf_hidden_ln)
    hf_v = hf_attn.v_proj(hf_hidden_ln)
    
    # Reshape
    input_shape = hf_hidden_ln.shape[:-1]
    hf_q = hf_q.view(*input_shape, -1, 64).transpose(1, 2)
    hf_k = hf_k.view(*input_shape, 2, 64).transpose(1, 2)
    hf_v = hf_v.view(*input_shape, 2, 64).transpose(1, 2)
    
    # RoPE
    rotary_emb = hf_model.model.rotary_emb
    k_for_rope = hf_k.transpose(1, 2)
    cos, sin = rotary_emb(k_for_rope, position_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)
    
    # Repeat K/V for GQA
    repeat_factor = 14 // 2
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v.repeat_interleave(repeat_factor, dim=1)
    
    # Attention
    scale = 1.0 / (64 ** 0.5)
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    
    print(f"\nHF Attention scores: {hf_scores.shape}, scale={scale:.6f}")
    print(f"  Scores range: [{hf_scores.min().item():.3f}, {hf_scores.max().item():.3f}]")
    
    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    hf_attn_weights = torch.softmax(hf_scores_masked, dim=-1)
    hf_attn_output = torch.matmul(hf_attn_weights, hf_v_repeated)
    
    print(f"HF Attention weights: {hf_attn_weights.shape}")
    print(f"  Weights range: [{hf_attn_weights.min().item():.6f}, {hf_attn_weights.max().item():.6f}]")
    print(f"  Weights sum (should be 1): {hf_attn_weights.sum(dim=-1).mean().item():.6f}")
    
    # Reshape HF attention output for comparison
    hf_attn_output_flat = hf_attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    
    # Apply o_proj for HF
    hf_final_output = hf_attn.o_proj(hf_attn_output_flat)
    
    # TP attention - call forward
    with torch.inference_mode():
        tp_attn_output, _ = tp_layer0.self_attn(
            hf_hidden_ln,
            kv_cache=None,
            position_ids=position_ids,
            rotary_emb=tp_model.rotary_emb,
        )
    
    # Compare (both should be [batch, seq_len, hidden_size])
    print(f"\nFinal attention output comparison (after o_proj):")
    print(f"  HF shape: {hf_final_output.shape}")
    print(f"  TP shape: {tp_attn_output.shape}")
    
    if hf_final_output.shape == tp_attn_output.shape:
        max_diff_attn = (hf_final_output - tp_attn_output).abs().max().item()
        mean_diff_attn = (hf_final_output - tp_attn_output).abs().mean().item()
        
        print(f"  Max diff: {max_diff_attn:.6f}")
        print(f"  Mean diff: {mean_diff_attn:.6f}", end="")
        print(" ✅" if max_diff_attn < 1e-2 else " ❌")
        
        # Check if it's a systematic difference or random
        if max_diff_attn > 1e-2:
            # Show where the max difference is
            diff = (hf_final_output - tp_attn_output).abs()
            max_idx = diff.argmax().item()
            flat_hf = hf_final_output.flatten()
            flat_tp = tp_attn_output.flatten()
            print(f"\n  Max diff at index {max_idx}: HF={flat_hf[max_idx]:.6f}, TP={flat_tp[max_idx]:.6f}")
    else:
        print(f"  ❌ Shape mismatch! Cannot compare.")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_attention_output()
