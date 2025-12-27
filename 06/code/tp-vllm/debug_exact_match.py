"""
Exact match debugging: Compare every single step
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def debug_exact_match():
    """Compare every step exactly"""
    print("="*70)
    print("Exact Match Debugging")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12381'
        dist.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo')
    
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
        attn_implementation="eager",
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
    
    # Use SAME input for both!
    same_input = hf_embeds.clone()
    
    hf_ln_out = hf_layer1.input_layernorm(same_input)
    tp_ln_out = tp_layer1.input_layernorm(same_input)
    
    print("\n1. Input LayerNorm (same input):")
    ln_diff = (hf_ln_out - tp_ln_out).abs().max().item()
    print(f"   Max diff: {ln_diff:.6f}", "✅" if ln_diff < 1e-5 else "❌")
    
    # Get attention modules
    hf_attn = hf_layer1.self_attn
    tp_attn = tp_layer1.self_attn
    
    # QKV projection
    print("\n2. QKV Projection (same input):")
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
    
    # Reshape
    batch_size, seq_len = hf_ln_out.shape[:2]
    hf_q = hf_q.view(batch_size, seq_len, -1, 64).transpose(1, 2)
    hf_k = hf_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    hf_v = hf_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    tp_q = tp_q.view(batch_size, seq_len, 14, 64).transpose(1, 2)
    tp_k = tp_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    tp_v = tp_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    # RoPE
    print("\n3. RoPE:")
    rotary_emb = hf_model.model.rotary_emb
    
    hf_k_for_rope = hf_k.transpose(1, 2)
    hf_cos, hf_sin = rotary_emb(hf_k_for_rope, position_ids)
    
    tp_k_for_rope = tp_k.transpose(1, 2)
    tp_cos, tp_sin = rotary_emb(tp_k_for_rope, position_ids)
    
    cos_diff = (hf_cos - tp_cos).abs().max().item()
    sin_diff = (hf_sin - tp_sin).abs().max().item()
    print(f"   cos diff: {cos_diff:.6f}", "✅" if cos_diff < 1e-5 else "❌")
    print(f"   sin diff: {sin_diff:.6f}", "✅" if sin_diff < 1e-5 else "❌")
    
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, hf_cos, hf_sin)
    tp_q_rope, tp_k_rope = apply_rotary_pos_emb(tp_q, tp_k, tp_cos, tp_sin)
    
    q_rope_diff = (hf_q_rope - tp_q_rope).abs().max().item()
    k_rope_diff = (hf_k_rope - tp_k_rope).abs().max().item()
    print(f"   Q after RoPE diff: {q_rope_diff:.6f}", "✅" if q_rope_diff < 1e-5 else "❌")
    print(f"   K after RoPE diff: {k_rope_diff:.6f}", "✅" if k_rope_diff < 1e-5 else "❌")
    
    # Repeat
    print("\n4. Repeat K/V:")
    repeat_factor = 14 // 2
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v.repeat_interleave(repeat_factor, dim=1)
    
    tp_k_repeated = tp_k_rope.repeat_interleave(repeat_factor, dim=1)
    tp_v_repeated = tp_v.repeat_interleave(repeat_factor, dim=1)
    
    k_rep_diff = (hf_k_repeated - tp_k_repeated).abs().max().item()
    v_rep_diff = (hf_v_repeated - tp_v_repeated).abs().max().item()
    print(f"   K repeated diff: {k_rep_diff:.6f}", "✅" if k_rep_diff < 1e-5 else "❌")
    print(f"   V repeated diff: {v_rep_diff:.6f}", "✅" if v_rep_diff < 1e-5 else "❌")
    
    # Scores
    print("\n5. Attention Scores:")
    scale = 1.0 / (64 ** 0.5)
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    tp_scores = torch.matmul(tp_q_rope, tp_k_repeated.transpose(-2, -1)) * scale
    
    scores_diff = (hf_scores - tp_scores).abs().max().item()
    print(f"   Scores diff: {scores_diff:.6f}", "✅" if scores_diff < 1e-5 else "❌")
    
    # Mask
    print("\n6. Causal Mask:")
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    tp_scores_masked = tp_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    scores_masked_diff = (hf_scores_masked - tp_scores_masked).abs().max().item()
    print(f"   Scores after mask diff: {scores_masked_diff:.6f}", "✅" if scores_masked_diff < 1e-5 else "❌")
    
    # Softmax
    print("\n7. Softmax:")
    hf_attn_weights = torch.softmax(hf_scores_masked, dim=-1, dtype=torch.float32).to(dtype)
    tp_attn_weights = torch.softmax(tp_scores_masked, dim=-1, dtype=torch.float32).to(dtype)
    
    weights_diff = (hf_attn_weights - tp_attn_weights).abs().max().item()
    print(f"   Attention weights diff: {weights_diff:.6f}", "✅" if weights_diff < 1e-5 else "❌")
    
    # Attention output
    print("\n8. Attention Output:")
    hf_attn_output = torch.matmul(hf_attn_weights, hf_v_repeated)
    tp_attn_output = torch.matmul(tp_attn_weights, tp_v_repeated)
    
    attn_out_diff = (hf_attn_output - tp_attn_output).abs().max().item()
    print(f"   Attention output diff: {attn_out_diff:.6f}", "✅" if attn_out_diff < 1e-5 else "❌")
    
    # Reshape and o_proj
    print("\n9. Reshape and o_proj:")
    hf_attn_output_reshaped = hf_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    tp_attn_output_reshaped = tp_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    hf_final = hf_attn.o_proj(hf_attn_output_reshaped)
    tp_final = tp_attn.o_proj(tp_attn_output_reshaped)
    
    final_diff = (hf_final - tp_final).abs().max().item()
    print(f"   Final output diff: {final_diff:.6f}", "✅" if final_diff < 1e-5 else "❌")
    
    # Now compare with TP forward
    print("\n10. TP Attention Forward:")
    tp_attn_output_forward, _ = tp_attn(
        tp_ln_out,
        kv_cache=None,
        position_ids=position_ids,
        rotary_emb=tp_model.rotary_emb,
    )
    
    forward_diff = (tp_final - tp_attn_output_forward).abs().max().item()
    print(f"   Manual vs Forward diff: {forward_diff:.6f}", "✅" if forward_diff < 1e-5 else "❌")
    
    # Compare HF manual with TP forward
    hf_tp_diff = (hf_final - tp_attn_output_forward).abs().max().item()
    print(f"   HF manual vs TP forward diff: {hf_tp_diff:.6f}", "✅" if hf_tp_diff < 1e-2 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    debug_exact_match()
