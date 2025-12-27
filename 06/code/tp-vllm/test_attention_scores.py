"""
Compare attention scores and weights in detail
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def test_attention_scores():
    """Compare attention scores step by step"""
    print("="*70)
    print("Attention Scores Comparison")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12361'
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
    
    # Create same input
    batch_size = 1
    seq_len = 3
    hidden_states = torch.randn(batch_size, seq_len, 896, dtype=dtype).to(device)
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    
    # Get layer 0
    hf_layer0 = hf_model.model.layers[0]
    tp_layer0 = tp_model.layers[0]
    
    # Apply layernorm
    hidden_ln = hf_layer0.input_layernorm(hidden_states)
    
    # Manually trace HF attention
    hf_attn = hf_layer0.self_attn
    hf_q = hf_attn.q_proj(hidden_ln)
    hf_k = hf_attn.k_proj(hidden_ln)
    hf_v = hf_attn.v_proj(hidden_ln)
    
    input_shape = hidden_ln.shape[:-1]
    hf_q = hf_q.view(*input_shape, -1, 64).transpose(1, 2)
    hf_k = hf_k.view(*input_shape, 2, 64).transpose(1, 2)
    hf_v = hf_v.view(*input_shape, 2, 64).transpose(1, 2)
    
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
    
    # Now get TP attention intermediate values
    # We need to modify attention.py to return intermediate values, or we can manually trace
    # For now, let's compare the final output and see if we can identify the issue
    
    # Get TP QKV output
    with torch.inference_mode():
        tp_qkv = tp_layer0.self_attn.qkv_proj(hidden_ln)
        tp_q, tp_k, tp_v = tp_layer0.self_attn.qkv_proj.split_qkv(tp_qkv)
        
        # Reshape
        tp_q = tp_q.view(batch_size, seq_len, 14, 64).transpose(1, 2)
        tp_k = tp_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
        tp_v = tp_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
        
        # Apply RoPE
        k_for_rope_tp = tp_k.transpose(1, 2)
        cos_tp, sin_tp = rotary_emb(k_for_rope_tp, position_ids)
        tp_q_rope, tp_k_rope = apply_rotary_pos_emb(tp_q, tp_k, cos_tp, sin_tp)
        
        # Repeat for GQA
        tp_k_repeated = tp_k_rope.repeat_interleave(repeat_factor, dim=1)
        tp_v_repeated = tp_v.repeat_interleave(repeat_factor, dim=1)
        
        # Attention scores
        tp_scores = torch.matmul(tp_q_rope, tp_k_repeated.transpose(-2, -1)) * scale
    
    # Compare scores
    print(f"\nAttention scores comparison:")
    print(f"  HF scores: {hf_scores.shape}")
    print(f"  TP scores: {tp_scores.shape}")
    
    if hf_scores.shape == tp_scores.shape:
        scores_diff = (hf_scores - tp_scores).abs()
        max_diff = scores_diff.max().item()
        mean_diff = scores_diff.mean().item()
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}", end="")
        print(" ✅" if max_diff < 1e-5 else " ❌")
        
        if max_diff > 1e-5:
            # Show where max diff is
            max_idx = scores_diff.argmax().item()
            flat_hf = hf_scores.flatten()
            flat_tp = tp_scores.flatten()
            print(f"  Max diff at index {max_idx}: HF={flat_hf[max_idx]:.6f}, TP={flat_tp[max_idx]:.6f}")
            
            # Compare Q, K after RoPE
            q_diff = (hf_q_rope - tp_q_rope).abs().max().item()
            k_diff = (hf_k_rope - tp_k_rope).abs().max().item()
            k_rep_diff = (hf_k_repeated - tp_k_repeated).abs().max().item()
            print(f"\n  Q after RoPE diff: {q_diff:.6f}", "✅" if q_diff < 1e-5 else "❌")
            print(f"  K after RoPE diff: {k_diff:.6f}", "✅" if k_diff < 1e-5 else "❌")
            print(f"  K repeated diff: {k_rep_diff:.6f}", "✅" if k_rep_diff < 1e-5 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_attention_scores()
