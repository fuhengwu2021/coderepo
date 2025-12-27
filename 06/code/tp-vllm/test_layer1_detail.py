"""
Detailed comparison of Layer 1 to find the issue
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def test_layer1_detail():
    """Compare Layer 1 in detail"""
    print("="*70)
    print("Layer 1 Detailed Comparison")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12372'
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
    
    # Get embeddings
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    tp_embeds = tp_model.embed_tokens(input_ids)
    
    # Get Layer 1
    hf_layer1 = hf_model.model.layers[0]
    tp_layer1 = tp_model.layers[0]
    
    # Compare input layernorm
    print("\n1. Input LayerNorm:")
    hf_ln_out = hf_layer1.input_layernorm(hf_embeds)
    tp_ln_out = tp_layer1.input_layernorm(tp_embeds)
    ln_diff = (hf_ln_out - tp_ln_out).abs().max().item()
    print(f"   Max diff: {ln_diff:.6f}", "✅" if ln_diff < 1e-5 else "❌")
    
    # Compare attention
    print("\n2. Attention:")
    # Get position embeddings for HF
    rotary_emb = hf_model.model.rotary_emb
    dummy_k = torch.zeros(1, input_ids.shape[1], 2, 64, dtype=dtype, device=device)
    cos, sin = rotary_emb(dummy_k, position_ids)
    position_embeddings = (cos, sin)
    
    # HF attention (needs attention_mask)
    attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.bool, device=device)
    hf_attn_out, _ = hf_layer1.self_attn(
        hf_ln_out,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    
    # TP attention
    tp_attn_out, _ = tp_layer1.self_attn(
        tp_ln_out,
        position_ids=position_ids,
        kv_cache=None,
        rotary_emb=tp_model.rotary_emb,
    )
    
    attn_diff = (hf_attn_out - tp_attn_out).abs().max().item()
    attn_mean_diff = (hf_attn_out - tp_attn_out).abs().mean().item()
    print(f"   Max diff: {attn_diff:.6f}, Mean diff: {attn_mean_diff:.6f}", 
          "✅" if attn_diff < 1e-2 else "❌")
    
    # Compare MLP
    print("\n3. MLP:")
    # HF: attention output goes through MLP
    hf_mlp_in = hf_ln_out + hf_attn_out  # Residual
    hf_mlp_ln = hf_layer1.post_attention_layernorm(hf_mlp_in)
    hf_mlp_out = hf_layer1.mlp(hf_mlp_ln)
    
    # TP: attention output goes through MLP
    tp_mlp_in = tp_ln_out + tp_attn_out  # Residual
    tp_mlp_ln = tp_layer1.post_attention_layernorm(tp_mlp_in)
    tp_mlp_out = tp_layer1.mlp(tp_mlp_ln)
    
    mlp_diff = (hf_mlp_out - tp_mlp_out).abs().max().item()
    mlp_mean_diff = (hf_mlp_out - tp_mlp_out).abs().mean().item()
    print(f"   Max diff: {mlp_diff:.6f}, Mean diff: {mlp_mean_diff:.6f}", 
          "✅" if mlp_diff < 1e-2 else "❌")
    
    # Compare final output
    print("\n4. Final Layer Output:")
    hf_final = hf_embeds + hf_attn_out + hf_mlp_out  # All residuals
    tp_final = tp_embeds + tp_attn_out + tp_mlp_out
    
    final_diff = (hf_final - tp_final).abs().max().item()
    final_mean_diff = (hf_final - tp_final).abs().mean().item()
    print(f"   Max diff: {final_diff:.6f}, Mean diff: {final_mean_diff:.6f}", 
          "✅" if final_diff < 1e-2 else "❌")
    
    # Compare with actual layer output
    print("\n5. Actual Layer Output:")
    hf_layer_out = hf_layer1(hf_embeds, position_ids=position_ids, position_embeddings=position_embeddings)
    tp_layer_out, _ = tp_layer1(tp_embeds, position_ids=position_ids, kv_cache=None)
    
    layer_diff = (hf_layer_out - tp_layer_out).abs().max().item()
    layer_mean_diff = (hf_layer_out - tp_layer_out).abs().mean().item()
    print(f"   Max diff: {layer_diff:.6f}, Mean diff: {layer_mean_diff:.6f}", 
          "✅" if layer_diff < 1e-2 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_layer1_detail()
