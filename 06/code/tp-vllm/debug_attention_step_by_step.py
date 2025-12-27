"""
Step-by-step attention debugging to find root cause
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def debug_attention_step_by_step():
    """Debug attention computation step by step"""
    print("="*70)
    print("Step-by-Step Attention Debugging")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12375'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create test input
    test_prompt = "What is the capital of France?"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
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
    
    # Get Layer 1
    hf_layer1 = hf_model.model.layers[0]
    tp_layer1 = tp_model.layers[0]
    
    # Get embeddings and apply layernorm
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    tp_embeds = tp_model.embed_tokens(input_ids)
    
    hf_ln_out = hf_layer1.input_layernorm(hf_embeds)
    tp_ln_out = tp_layer1.input_layernorm(tp_embeds)
    
    print("\n1. Input LayerNorm:")
    ln_diff = (hf_ln_out - tp_ln_out).abs().max().item()
    print(f"   Max diff: {ln_diff:.6f}", "✅" if ln_diff < 1e-5 else "❌")
    
    # Get attention modules
    hf_attn = hf_layer1.self_attn
    tp_attn = tp_layer1.self_attn
    
    # Step 1: QKV projection
    print("\n2. QKV Projection:")
    hf_q = hf_attn.q_proj(hf_ln_out)
    hf_k = hf_attn.k_proj(hf_ln_out)
    hf_v = hf_attn.v_proj(hf_ln_out)
    
    tp_qkv = tp_attn.qkv_proj(tp_ln_out)
    tp_q, tp_k, tp_v = tp_attn.qkv_proj.split_qkv(tp_qkv)
    
    q_diff = (hf_q - tp_q).abs().max().item()
    k_diff = (hf_k - tp_k).abs().max().item()
    v_diff = (hf_v - tp_v).abs().max().item()
    print(f"   Q diff: {q_diff:.6f}", "✅" if q_diff < 1e-5 else "❌")
    print(f"   K diff: {k_diff:.6f}", "✅" if k_diff < 1e-5 else "❌")
    print(f"   V diff: {v_diff:.6f}", "✅" if v_diff < 1e-5 else "❌")
    
    # Step 2: Reshape
    print("\n3. Reshape to [batch, num_heads, seq_len, head_dim]:")
    batch_size, seq_len = hf_ln_out.shape[:2]
    
    hf_q_reshaped = hf_q.view(batch_size, seq_len, -1, 64).transpose(1, 2)
    hf_k_reshaped = hf_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    hf_v_reshaped = hf_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    tp_q_reshaped = tp_q.view(batch_size, seq_len, 14, 64).transpose(1, 2)
    tp_k_reshaped = tp_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    tp_v_reshaped = tp_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    print(f"   HF shapes: q={hf_q_reshaped.shape}, k={hf_k_reshaped.shape}, v={hf_v_reshaped.shape}")
    print(f"   TP shapes: q={tp_q_reshaped.shape}, k={tp_k_reshaped.shape}, v={tp_v_reshaped.shape}")
    
    # Step 3: RoPE
    print("\n4. RoPE Application:")
    rotary_emb = hf_model.model.rotary_emb
    
    # HF RoPE
    hf_k_for_rope = hf_k_reshaped.transpose(1, 2)
    hf_cos, hf_sin = rotary_emb(hf_k_for_rope, position_ids)
    
    # TP RoPE
    tp_k_for_rope = tp_k_reshaped.transpose(1, 2)
    tp_cos, tp_sin = rotary_emb(tp_k_for_rope, position_ids)
    
    cos_diff = (hf_cos - tp_cos).abs().max().item()
    sin_diff = (hf_sin - tp_sin).abs().max().item()
    print(f"   cos diff: {cos_diff:.6f}", "✅" if cos_diff < 1e-5 else "❌")
    print(f"   sin diff: {sin_diff:.6f}", "✅" if sin_diff < 1e-5 else "❌")
    
    # Apply RoPE
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q_reshaped, hf_k_reshaped, hf_cos, hf_sin)
    tp_q_rope, tp_k_rope = apply_rotary_pos_emb(tp_q_reshaped, tp_k_reshaped, tp_cos, tp_sin)
    
    q_rope_diff = (hf_q_rope - tp_q_rope).abs().max().item()
    k_rope_diff = (hf_k_rope - tp_k_rope).abs().max().item()
    print(f"   Q after RoPE diff: {q_rope_diff:.6f}", "✅" if q_rope_diff < 1e-5 else "❌")
    print(f"   K after RoPE diff: {k_rope_diff:.6f}", "✅" if k_rope_diff < 1e-5 else "❌")
    
    # Step 4: Repeat K/V for GQA
    print("\n5. Repeat K/V for GQA:")
    repeat_factor = 14 // 2
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v_reshaped.repeat_interleave(repeat_factor, dim=1)
    
    tp_k_repeated = tp_k_rope.repeat_interleave(repeat_factor, dim=1)
    tp_v_repeated = tp_v_reshaped.repeat_interleave(repeat_factor, dim=1)
    
    k_rep_diff = (hf_k_repeated - tp_k_repeated).abs().max().item()
    v_rep_diff = (hf_v_repeated - tp_v_repeated).abs().max().item()
    print(f"   K repeated diff: {k_rep_diff:.6f}", "✅" if k_rep_diff < 1e-5 else "❌")
    print(f"   V repeated diff: {v_rep_diff:.6f}", "✅" if v_rep_diff < 1e-5 else "❌")
    
    # Step 5: Attention scores
    print("\n6. Attention Scores:")
    scale = 1.0 / (64 ** 0.5)
    print(f"   Scale: {scale:.6f}")
    
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    tp_scores = torch.matmul(tp_q_rope, tp_k_repeated.transpose(-2, -1)) * scale
    
    scores_diff = (hf_scores - tp_scores).abs()
    max_scores_diff = scores_diff.max().item()
    mean_scores_diff = scores_diff.mean().item()
    
    print(f"   Scores shape: {hf_scores.shape}")
    print(f"   Scores range: HF=[{hf_scores.min().item():.3f}, {hf_scores.max().item():.3f}], "
          f"TP=[{tp_scores.min().item():.3f}, {tp_scores.max().item():.3f}]")
    print(f"   Max diff: {max_scores_diff:.6f}, Mean diff: {mean_scores_diff:.6f}", 
          "✅" if max_scores_diff < 1e-3 else "❌")
    
    # Step 6: Causal mask
    print("\n7. Causal Mask:")
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    # Check how TP does it
    # TP should do the same in attention.py
    
    # Check for NaN
    hf_has_nan = torch.isnan(hf_scores_masked).any().item()
    tp_has_nan = torch.isnan(tp_scores).any().item()
    print(f"   HF has NaN: {hf_has_nan}")
    print(f"   TP has NaN: {tp_has_nan}")
    
    # Step 7: Softmax
    print("\n8. Softmax:")
    hf_attn_weights = torch.softmax(hf_scores_masked, dim=-1)
    
    # We need to check TP's softmax - let's manually compute it
    # But first, let's see if TP applies mask correctly
    # Actually, let's call TP attention and compare outputs
    
    # Step 8: Attention output
    print("\n9. Attention Output (Q @ K^T @ V):")
    hf_attn_output = torch.matmul(hf_attn_weights, hf_v_repeated)
    
    # Get TP attention output
    tp_attn_output, _ = tp_attn(
        tp_ln_out,
        kv_cache=None,
        position_ids=position_ids,
        rotary_emb=tp_model.rotary_emb,
    )
    
    # Reshape TP output for comparison
    tp_attn_output_reshaped = tp_attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    hf_attn_output_reshaped = hf_attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    
    attn_out_diff = (hf_attn_output_reshaped - tp_attn_output_reshaped).abs()
    max_attn_diff = attn_out_diff.max().item()
    mean_attn_diff = attn_out_diff.mean().item()
    
    print(f"   Max diff: {max_attn_diff:.6f}, Mean diff: {mean_attn_diff:.6f}", 
          "✅" if max_attn_diff < 1e-2 else "❌")
    
    # Step 9: Output projection
    print("\n10. Output Projection (o_proj):")
    hf_final = hf_attn.o_proj(hf_attn_output_reshaped)
    tp_final = tp_attn.o_proj(tp_attn_output)
    
    final_diff = (hf_final - tp_final).abs()
    max_final_diff = final_diff.max().item()
    mean_final_diff = final_diff.mean().item()
    
    print(f"   Max diff: {max_final_diff:.6f}, Mean diff: {mean_final_diff:.6f}", 
          "✅" if max_final_diff < 1e-2 else "❌")
    
    # Find where the difference starts
    print("\n" + "="*70)
    print("Difference Analysis:")
    print("="*70)
    
    steps = [
        ("QKV Projection", max(q_diff, k_diff, v_diff)),
        ("After RoPE", max(q_rope_diff, k_rope_diff)),
        ("After Repeat", max(k_rep_diff, v_rep_diff)),
        ("Attention Scores", max_scores_diff),
        ("Attention Output", max_attn_diff),
        ("Final Output", max_final_diff),
    ]
    
    print("\nDifference at each step:")
    for step_name, diff in steps:
        status = "✅" if diff < 1e-3 else "⚠️" if diff < 1.0 else "❌"
        print(f"  {step_name:20s}: {diff:10.6f} {status}")
    
    # Check if there's a sudden jump
    print("\nDifference growth:")
    prev_diff = 0
    for step_name, diff in steps:
        growth = diff - prev_diff
        if growth > 0.1:
            print(f"  ⚠️ Large jump at {step_name}: +{growth:.6f}")
        prev_diff = diff
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    debug_attention_step_by_step()
