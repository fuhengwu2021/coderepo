"""
Debug softmax computation specifically
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def debug_softmax():
    """Debug softmax computation"""
    print("="*70)
    print("Softmax Debugging")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12376'
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
    
    # Get attention modules
    hf_attn = hf_layer1.self_attn
    tp_attn = tp_layer1.self_attn
    
    # QKV projection and reshape
    hf_q = hf_attn.q_proj(hf_ln_out)
    hf_k = hf_attn.k_proj(hf_ln_out)
    hf_v = hf_attn.v_proj(hf_ln_out)
    
    batch_size, seq_len = hf_ln_out.shape[:2]
    hf_q = hf_q.view(batch_size, seq_len, -1, 64).transpose(1, 2)
    hf_k = hf_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    hf_v = hf_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    # RoPE
    rotary_emb = hf_model.model.rotary_emb
    k_for_rope = hf_k.transpose(1, 2)
    cos, sin = rotary_emb(k_for_rope, position_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)
    
    # Repeat for GQA
    repeat_factor = 14 // 2
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v.repeat_interleave(repeat_factor, dim=1)
    
    # Attention scores
    scale = 1.0 / (64 ** 0.5)
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    
    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    # Softmax
    print("\n1. Softmax Computation:")
    hf_attn_weights = torch.softmax(hf_scores_masked, dim=-1)
    
    print(f"   Scores shape: {hf_scores_masked.shape}")
    print(f"   Scores range: [{hf_scores_masked.min().item():.3f}, {hf_scores_masked.max().item():.3f}]")
    print(f"   Scores has -inf: {(hf_scores_masked == float('-inf')).any().item()}")
    print(f"   Scores has NaN: {torch.isnan(hf_scores_masked).any().item()}")
    
    print(f"\n   Attention weights shape: {hf_attn_weights.shape}")
    print(f"   Attention weights range: [{hf_attn_weights.min().item():.6f}, {hf_attn_weights.max().item():.6f}]")
    print(f"   Attention weights sum (should be 1): {hf_attn_weights.sum(dim=-1).mean().item():.6f}")
    print(f"   Attention weights has NaN: {torch.isnan(hf_attn_weights).any().item()}")
    
    # Now get TP attention weights
    print("\n2. TP Attention Weights:")
    # We need to extract from TP attention forward
    # Let's modify to return intermediate values or trace through
    
    # Actually, let's manually compute what TP should do
    # Get TP QKV
    tp_qkv = tp_attn.qkv_proj(tp_ln_out)
    tp_q, tp_k, tp_v = tp_attn.qkv_proj.split_qkv(tp_qkv)
    
    tp_q = tp_q.view(batch_size, seq_len, 14, 64).transpose(1, 2)
    tp_k = tp_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    tp_v = tp_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    # TP RoPE
    tp_k_for_rope = tp_k.transpose(1, 2)
    tp_cos, tp_sin = rotary_emb(tp_k_for_rope, position_ids)
    tp_q_rope, tp_k_rope = apply_rotary_pos_emb(tp_q, tp_k, tp_cos, tp_sin)
    
    # TP Repeat
    tp_k_repeated = tp_k_rope.repeat_interleave(repeat_factor, dim=1)
    tp_v_repeated = tp_v.repeat_interleave(repeat_factor, dim=1)
    
    # TP Scores
    tp_scores = torch.matmul(tp_q_rope, tp_k_repeated.transpose(-2, -1)) * scale
    
    # Check if scores match
    scores_diff = (hf_scores_masked - tp_scores).abs().max().item()
    print(f"   Scores diff (before mask): {(hf_scores - tp_scores).abs().max().item():.6f}")
    
    # TP mask - check how it's applied
    # Let's check attention.py to see how mask is applied
    print(f"\n3. Checking TP Mask Application:")
    # Read attention.py to see mask logic
    
    # For now, let's apply mask the same way
    tp_scores_masked = tp_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    scores_masked_diff = (hf_scores_masked - tp_scores_masked).abs().max().item()
    print(f"   Scores after mask diff: {scores_masked_diff:.6f}", 
          "✅" if scores_masked_diff < 1e-5 else "❌")
    
    # TP Softmax
    tp_attn_weights = torch.softmax(tp_scores_masked, dim=-1)
    
    # Compare weights
    weights_diff = (hf_attn_weights - tp_attn_weights).abs()
    max_weights_diff = weights_diff.max().item()
    mean_weights_diff = weights_diff.mean().item()
    
    print(f"\n4. Attention Weights Comparison:")
    print(f"   Max diff: {max_weights_diff:.6f}, Mean diff: {mean_weights_diff:.6f}", 
          "✅" if max_weights_diff < 1e-5 else "❌")
    
    if max_weights_diff > 1e-5:
        # Find where max diff is
        max_idx = weights_diff.argmax().item()
        flat_hf = hf_attn_weights.flatten()
        flat_tp = tp_attn_weights.flatten()
        print(f"   Max diff at index {max_idx}: HF={flat_hf[max_idx]:.6f}, TP={flat_tp[max_idx]:.6f}")
        
        # Check corresponding scores
        flat_hf_scores = hf_scores_masked.flatten()
        flat_tp_scores = tp_scores_masked.flatten()
        print(f"   Corresponding scores: HF={flat_hf_scores[max_idx]:.6f}, TP={flat_tp_scores[max_idx]:.6f}")
    
    # Attention output
    print(f"\n5. Attention Output:")
    hf_attn_output = torch.matmul(hf_attn_weights, hf_v_repeated)
    tp_attn_output = torch.matmul(tp_attn_weights, tp_v_repeated)
    
    attn_out_diff = (hf_attn_output - tp_attn_output).abs()
    max_attn_out_diff = attn_out_diff.max().item()
    mean_attn_out_diff = attn_out_diff.mean().item()
    
    print(f"   Max diff: {max_attn_out_diff:.6f}, Mean diff: {mean_attn_out_diff:.6f}", 
          "✅" if max_attn_out_diff < 1e-3 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    debug_softmax()
