"""
Debug attention weights specifically to find the 0.4 difference
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def debug_attention_weights():
    """Debug attention weights in detail"""
    print("="*70)
    print("Attention Weights Detailed Debugging")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12380'
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
    
    hf_ln_out = hf_layer1.input_layernorm(hf_embeds)
    tp_ln_out = tp_layer1.input_layernorm(tp_embeds)
    
    # Get attention modules
    hf_attn = hf_layer1.self_attn
    tp_attn = tp_layer1.self_attn
    
    # Manual computation for both
    print("\n1. Manual Computation (HF):")
    
    # HF manual
    hf_q = hf_attn.q_proj(hf_ln_out)
    hf_k = hf_attn.k_proj(hf_ln_out)
    hf_v = hf_attn.v_proj(hf_ln_out)
    
    batch_size, seq_len = hf_ln_out.shape[:2]
    hf_q = hf_q.view(batch_size, seq_len, -1, 64).transpose(1, 2)
    hf_k = hf_k.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    hf_v = hf_v.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    
    rotary_emb = hf_model.model.rotary_emb
    k_for_rope = hf_k.transpose(1, 2)
    cos, sin = rotary_emb(k_for_rope, position_ids)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)
    
    repeat_factor = 14 // 2
    hf_k_repeated = hf_k_rope.repeat_interleave(repeat_factor, dim=1)
    hf_v_repeated = hf_v.repeat_interleave(repeat_factor, dim=1)
    
    scale = 1.0 / (64 ** 0.5)
    hf_scores = torch.matmul(hf_q_rope, hf_k_repeated.transpose(-2, -1)) * scale
    
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    hf_scores_masked = hf_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
    hf_attn_weights_manual = torch.softmax(hf_scores_masked, dim=-1, dtype=torch.float32).to(dtype)
    hf_attn_output_manual = torch.matmul(hf_attn_weights_manual, hf_v_repeated)
    
    print("\n2. TP Attention Forward:")
    tp_attn_output_forward, _ = tp_attn(
        tp_ln_out,
        kv_cache=None,
        position_ids=position_ids,
        rotary_emb=tp_model.rotary_emb,
    )
    
    print("\n3. Comparison:")
    # Reshape for comparison
    hf_attn_output_manual_reshaped = hf_attn_output_manual.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    hf_attn_output_manual_final = hf_attn.o_proj(hf_attn_output_manual_reshaped)
    
    diff = (hf_attn_output_manual_final - tp_attn_output_forward).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"   HF manual vs TP forward:")
    print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}", 
          "✅" if max_diff < 1e-3 else "❌")
    
    # Also compare with HF actual forward
    print("\n4. HF Actual Forward:")
    # Get position embeddings
    dummy_k = torch.zeros(1, seq_len, 2, 64, dtype=dtype, device=device)
    cos, sin = rotary_emb(dummy_k, position_ids)
    position_embeddings = (cos, sin)
    
    attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    hf_attn_output_actual, _ = hf_attn(
        hf_ln_out,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    
    diff_actual = (hf_attn_output_actual - tp_attn_output_forward).abs()
    max_diff_actual = diff_actual.max().item()
    mean_diff_actual = diff_actual.mean().item()
    
    print(f"   HF actual vs TP forward:")
    print(f"   Max diff: {max_diff_actual:.6f}, Mean diff: {mean_diff_actual:.6f}", 
          "✅" if max_diff_actual < 1e-2 else "❌")
    
    # Compare manual vs actual
    diff_manual_actual = (hf_attn_output_manual_final - hf_attn_output_actual).abs()
    max_diff_manual_actual = diff_manual_actual.max().item()
    print(f"\n   HF manual vs HF actual:")
    print(f"   Max diff: {max_diff_manual_actual:.6f}", 
          "✅" if max_diff_manual_actual < 1e-3 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    debug_attention_weights()
