"""
Layer-by-Layer Comparison: Find where the difference starts
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
from parallel_state import initialize_tensor_parallel

def test_layer_by_layer():
    """Compare each layer's output to find where difference starts"""
    print("="*70)
    print("Layer-by-Layer Comparison")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize
    if not dist.is_initialized():
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12371'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create test input
    test_prompt = "What is the capital of France?"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    print(f"\nTest prompt: {test_prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    # Load models
    print("\nLoading models...")
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
    
    # Compare embeddings
    print("\n" + "="*70)
    print("Comparing Embeddings:")
    print("="*70)
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    tp_embeds = tp_model.embed_tokens(input_ids)
    embed_diff = (hf_embeds - tp_embeds).abs().max().item()
    print(f"Embeddings: Max diff={embed_diff:.6f}", "✅" if embed_diff < 1e-5 else "❌")
    
    # Compare each layer
    print("\n" + "="*70)
    print("Comparing Each Layer:")
    print("="*70)
    
    hf_hidden = hf_embeds
    tp_hidden = tp_embeds
    
    # Get position embeddings for HF (needed for Qwen2)
    rotary_emb = hf_model.model.rotary_emb
    # Create dummy k for getting cos/sin (HF needs this format)
    dummy_k = torch.zeros(1, input_ids.shape[1], 2, 64, dtype=dtype, device=device)
    cos, sin = rotary_emb(dummy_k, position_ids)
    position_embeddings = (cos, sin)
    
    for layer_idx in range(hf_model.config.num_hidden_layers):
        # HF layer (needs position_embeddings tuple, not position_ids)
        hf_layer = hf_model.model.layers[layer_idx]
        hf_hidden = hf_layer(
            hf_hidden, 
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        
        # TP layer (returns tuple: hidden_states, kv_cache)
        tp_layer = tp_model.layers[layer_idx]
        tp_hidden, _ = tp_layer(
            tp_hidden,
            position_ids=position_ids,
            kv_cache=None,
        )
        
        # Compare
        layer_diff = (hf_hidden - tp_hidden).abs().max().item()
        layer_mean_diff = (hf_hidden - tp_hidden).abs().mean().item()
        
        status = "✅" if layer_diff < 1e-3 else "⚠️" if layer_diff < 1.0 else "❌"
        print(f"Layer {layer_idx+1:2d}: Max diff={layer_diff:8.6f}, Mean diff={layer_mean_diff:8.6f} {status}")
        
        # If difference is large, show more details
        if layer_diff > 1.0:
            print(f"  ⚠️ Large difference detected at layer {layer_idx+1}!")
            # Show where max diff is
            diff_tensor = (hf_hidden - tp_hidden).abs()
            max_idx = diff_tensor.argmax().item()
            flat_hf = hf_hidden.flatten()
            flat_tp = tp_hidden.flatten()
            print(f"  Max diff at index {max_idx}: HF={flat_hf[max_idx]:.6f}, TP={flat_tp[max_idx]:.6f}")
    
    # Compare final layer norm
    print("\n" + "="*70)
    print("Comparing Final Layer Norm:")
    print("="*70)
    hf_final = hf_model.model.norm(hf_hidden)
    tp_final = tp_model.norm(tp_hidden)
    norm_diff = (hf_final - tp_final).abs().max().item()
    print(f"Final norm: Max diff={norm_diff:.6f}", "✅" if norm_diff < 1e-3 else "❌")
    
    # Compare final logits
    print("\n" + "="*70)
    print("Comparing Final Logits:")
    print("="*70)
    hf_logits = hf_model.lm_head(hf_final)
    tp_logits = tp_model.lm_head(tp_final)
    logits_diff = (hf_logits - tp_logits).abs().max().item()
    logits_mean_diff = (hf_logits - tp_logits).abs().mean().item()
    print(f"Final logits: Max diff={logits_diff:.6f}, Mean diff={logits_mean_diff:.6f}", 
          "✅" if logits_diff < 1e-2 else "❌")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    test_layer_by_layer()
