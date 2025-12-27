"""
Compare TP attention forward vs manual computation
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def debug_attention_forward():
    """Compare TP attention forward vs manual"""
    print("="*70)
    print("TP Attention Forward vs Manual Computation")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12377'
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
    
    # Get TP attention
    tp_attn = tp_layer1.self_attn
    
    print("\n1. Calling TP Attention Forward:")
    tp_attn_output_forward, _ = tp_attn(
        tp_ln_out,
        kv_cache=None,
        position_ids=position_ids,
        rotary_emb=tp_model.rotary_emb,
    )
    
    print(f"   TP forward output shape: {tp_attn_output_forward.shape}")
    
    # Manual computation
    print("\n2. Manual Computation:")
    
    # QKV
    tp_qkv = tp_attn.qkv_proj(tp_ln_out)
    tp_q, tp_k, tp_v = tp_attn.qkv_proj.split_qkv(tp_qkv)
    
    batch_size, seq_len = tp_ln_out.shape[:2]
    tp_q = tp_q.view(batch_size, seq_len, 14, 64).transpose(1, 2)
    tp_k = tp_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    tp_v = tp_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    # RoPE
    rotary_emb = tp_model.rotary_emb
    tp_k_for_rope = tp_k.transpose(1, 2)
    tp_cos, tp_sin = rotary_emb(tp_k_for_rope, position_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    tp_q_rope, tp_k_rope = apply_rotary_pos_emb(tp_q, tp_k, tp_cos, tp_sin)
    
    # Repeat
    repeat_factor = 14 // 2
    tp_k_repeated = tp_k_rope.repeat_interleave(repeat_factor, dim=1)
    tp_v_repeated = tp_v.repeat_interleave(repeat_factor, dim=1)
    
    # Scores
    scale = 1.0 / (64 ** 0.5)
    tp_scores = torch.matmul(tp_q_rope, tp_k_repeated.transpose(-2, -1)) * scale
    
    # Mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    tp_scores_masked = tp_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    # Softmax
    tp_attn_weights = torch.softmax(tp_scores_masked, dim=-1)
    
    # Output
    tp_attn_output_manual = torch.matmul(tp_attn_weights, tp_v_repeated)
    
    # Reshape
    tp_attn_output_manual_reshaped = tp_attn_output_manual.transpose(1, 2).contiguous()
    tp_attn_output_manual_reshaped = tp_attn_output_manual_reshaped.view(batch_size, seq_len, -1)
    
    # o_proj
    tp_attn_output_manual_final = tp_attn.o_proj(tp_attn_output_manual_reshaped)
    
    print(f"   Manual output shape: {tp_attn_output_manual_final.shape}")
    
    # Compare
    print("\n3. Comparison:")
    diff = (tp_attn_output_forward - tp_attn_output_manual_final).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}", 
          "✅" if max_diff < 1e-5 else "❌")
    
    if max_diff > 1e-5:
        # Check intermediate steps
        print("\n4. Checking Intermediate Steps:")
        
        # Check before o_proj
        tp_attn_output_before_o_proj_forward = tp_attn_output_forward  # Actually, we need to extract this
        # Let's check the shapes and see if reshape is the issue
        
        # Actually, let's check if the issue is in the forward method itself
        # Let's trace through the forward method step by step
        
        print("   Need to check TP attention forward implementation...")
        
        # Check if the issue is with how we're calling it
        # Maybe the issue is with the input format or something else
    
    # Also compare with HF
    print("\n5. Comparison with HF:")
    hf_attn = hf_layer1.self_attn
    
    # Get position embeddings for HF
    hf_k_for_rope = hf_k_reshaped.transpose(1, 2) if 'hf_k_reshaped' in locals() else None
    if hf_k_for_rope is None:
        hf_q = hf_attn.q_proj(hf_ln_out)
        hf_k = hf_attn.k_proj(hf_ln_out)
        hf_v = hf_attn.v_proj(hf_ln_out)
        hf_q = hf_q.view(batch_size, seq_len, -1, 64).transpose(1, 2)
        hf_k = hf_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
        hf_v = hf_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
        hf_k_for_rope = hf_k.transpose(1, 2)
    
    hf_cos, hf_sin = rotary_emb(hf_k_for_rope, position_ids)
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, hf_cos, hf_sin)
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v.repeat_interleave(repeat_factor, dim=1)
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    hf_attn_weights = torch.softmax(hf_scores_masked, dim=-1)
    hf_attn_output = torch.matmul(hf_attn_weights, hf_v_repeated)
    hf_attn_output_reshaped = hf_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    hf_attn_output_final = hf_attn.o_proj(hf_attn_output_reshaped)
    
    hf_diff = (hf_attn_output_final - tp_attn_output_forward).abs()
    hf_max_diff = hf_diff.max().item()
    hf_mean_diff = hf_diff.mean().item()
    
    print(f"   HF vs TP forward: Max diff={hf_max_diff:.6f}, Mean diff={hf_mean_diff:.6f}", 
          "✅" if hf_max_diff < 1e-2 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    debug_attention_forward()
