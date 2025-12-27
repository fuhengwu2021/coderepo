"""
Detailed Comparison: Compare intermediate outputs between HF and TP model

This helps identify where the difference occurs.
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel

def compare_tensors(name: str, tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-5):
    """Compare two tensors and report differences"""
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name}: Shape mismatch! {tensor1.shape} vs {tensor2.shape}")
        return False
    
    max_diff = (tensor1 - tensor2).abs().max().item()
    mean_diff = (tensor1 - tensor2).abs().mean().item()
    rel_diff = (tensor1 - tensor2).abs() / (tensor2.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✅ {name}: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f}, Max rel diff={max_rel_diff:.6f}")
    else:
        print(f"❌ {name}: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f}, Max rel diff={max_rel_diff:.6f} (NOT CLOSE)")
        # Show where the max difference occurs
        max_idx = (tensor1 - tensor2).abs().argmax().item()
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()
        print(f"   Max diff at index {max_idx}: TP={flat1[max_idx]:.6f}, HF={flat2[max_idx]:.6f}")
    
    return is_close

def test_detailed_comparison():
    """Compare HF and TP model at multiple points"""
    print("="*70)
    print("Detailed Comparison: HF vs TP Model (tp=1)")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize distributed for TP
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    from parallel_state import initialize_tensor_parallel
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create test input
    test_prompt = "Hello"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    print(f"\nTest prompt: {test_prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    # Load HF original model
    print("\nLoading HuggingFace original model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=str(device),
        dtype=dtype,
    )
    hf_model.eval()
    
    # Create HybridTPModel (tp=1)
    print("Creating HybridTPModel (tp=1)...")
    tp_model = HybridTPModel(
        model_name=model_name,
        device=str(device),
        dtype=dtype,
    )
    tp_model.eval()
    
    # Compare embeddings
    print("\n" + "="*70)
    print("1. Comparing Embeddings:")
    print("="*70)
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    tp_embeds = tp_model.embed_tokens(input_ids)
    compare_tensors("Embeddings", tp_embeds, hf_embeds)
    
    # Compare first layer
    print("\n" + "="*70)
    print("2. Comparing First Layer:")
    print("="*70)
    
    # HF first layer
    hf_hidden = hf_embeds
    hf_layer0 = hf_model.model.layers[0]
    hf_hidden_after_ln = hf_layer0.input_layernorm(hf_hidden)
    
    # TP first layer
    tp_hidden = tp_embeds
    tp_layer0 = tp_model.layers[0]
    tp_hidden_after_ln = tp_layer0.input_layernorm(tp_hidden)
    compare_tensors("After input_layernorm", tp_hidden_after_ln, hf_hidden_after_ln)
    
    # Compare attention output (this is where RoPE is applied)
    # For HF, we need to manually trace through attention
    print("\n3. Comparing Attention Output (Layer 0):")
    print("="*70)
    
    # HF attention manually
    hf_attn = hf_layer0.self_attn
    hf_q = hf_attn.q_proj(hf_hidden_after_ln)
    hf_k = hf_attn.k_proj(hf_hidden_after_ln)
    hf_v = hf_attn.v_proj(hf_hidden_after_ln)
    
    # Reshape and transpose to [batch, num_heads, seq_len, head_dim]
    hf_q = hf_q.view(1, -1, hf_q.shape[1], 64).transpose(1, 2)
    hf_k = hf_k.view(1, -1, hf_k.shape[1], 64).transpose(1, 2)
    hf_v = hf_v.view(1, -1, hf_v.shape[1], 64).transpose(1, 2)
    
    # Get cos, sin
    rotary_emb = hf_model.model.rotary_emb
    k_for_rope = hf_k.transpose(1, 2)  # [batch, seq_len, num_kv_heads, head_dim]
    cos, sin = rotary_emb(k_for_rope, position_ids)
    
    # Apply RoPE
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q, hf_k, cos, sin)
    
    # TP attention - we need to extract intermediate outputs
    # This is tricky without modifying the code, but let's try to compare final outputs
    
    # Compare final logits
    print("\n" + "="*70)
    print("4. Comparing Final Logits:")
    print("="*70)
    with torch.inference_mode():
        hf_outputs = hf_model(input_ids, position_ids=position_ids)
        hf_logits = hf_outputs.logits
        
        tp_logits, _ = tp_model.forward(input_ids, position_ids=position_ids)
    
    compare_tensors("Final Logits", tp_logits, hf_logits, rtol=1e-1, atol=1e-2)
    
    # Check top-k tokens
    print("\n5. Comparing Top-5 Tokens:")
    print("="*70)
    hf_top5 = torch.topk(hf_logits[0, -1, :], 5)
    tp_top5 = torch.topk(tp_logits[0, -1, :], 5)
    
    print(f"HF top-5: {hf_top5.indices.tolist()}")
    print(f"TP top-5: {tp_top5.indices.tolist()}")
    print(f"Match: {torch.equal(hf_top5.indices, tp_top5.indices)}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_detailed_comparison()
