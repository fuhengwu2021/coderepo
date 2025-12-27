"""
Golden Check: Compare HybridTPModel (tp=1) with HuggingFace original model

This script verifies that our TP implementation produces the same results
as the original HuggingFace model when TP=1 (no parallelism).

Usage:
    python test_golden_check.py
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_tp_model import HybridTPModel
import numpy as np

def compare_tensors(name: str, tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-5):
    """Compare two tensors and report differences"""
    if tensor1.shape != tensor2.shape:
        print(f"❌ {name}: Shape mismatch! {tensor1.shape} vs {tensor2.shape}")
        return False
    
    max_diff = (tensor1 - tensor2).abs().max().item()
    mean_diff = (tensor1 - tensor2).abs().mean().item()
    
    # Check if tensors are close
    is_close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"✅ {name}: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f}")
    else:
        print(f"❌ {name}: Max diff={max_diff:.6f}, Mean diff={mean_diff:.6f} (NOT CLOSE)")
        # Show some sample values
        print(f"   Sample values: TP={tensor1.flatten()[:5].tolist()}, HF={tensor2.flatten()[:5].tolist()}")
    
    return is_close

def test_golden_check():
    """Test that HybridTPModel (tp=1) matches HF model"""
    print("="*70)
    print("Golden Check: HybridTPModel (tp=1) vs HuggingFace Original")
    print("="*70)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    
    # Initialize TP with size=1 (no parallelism)
    # For tp=1, we can run without distributed initialization
    # But HybridTPModel expects TP to be initialized, so we need to mock it
    # Check if distributed is already initialized
    if not dist.is_initialized():
        # Initialize with single process
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='gloo' if device.type == 'cpu' else 'nccl', 
                              init_method='env://', rank=0, world_size=1)
    
    from parallel_state import initialize_tensor_parallel
    initialize_tensor_parallel(tensor_parallel_size=1, backend='gloo' if device.type == 'cpu' else 'nccl')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create test input
    test_prompt = "What is the capital of France?"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    print(f"\nTest prompt: {test_prompt}")
    print(f"Input shape: {input_ids.shape}")
    
    # Load HF original model
    # CRITICAL: Use eager attention to match our TP implementation
    # sdpa (scaled dot product attention) uses optimized kernels that may have
    # different numerical precision, leading to larger differences
    print("\nLoading HuggingFace original model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=str(device),
        dtype=dtype,
        torch_dtype=dtype,
        attn_implementation="eager",  # Use eager to match TP implementation
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
    
    # Forward pass with both models
    print("\nRunning forward pass...")
    with torch.inference_mode():
        # HF model
        hf_outputs = hf_model(input_ids, position_ids=position_ids)
        hf_logits = hf_outputs.logits
        
        # TP model
        tp_logits, _ = tp_model.forward(input_ids, position_ids=position_ids)
    
    # Compare final logits
    print("\n" + "="*70)
    print("Comparing Final Logits:")
    print("="*70)
    logits_match = compare_tensors("Final Logits", tp_logits, hf_logits, rtol=1e-2, atol=1e-4)
    
    # Compare layer-by-layer (if possible)
    print("\n" + "="*70)
    print("Layer-by-Layer Comparison (if available):")
    print("="*70)
    
    # Test first layer attention output
    # This requires accessing intermediate outputs, which we can do by modifying forward temporarily
    # For now, we'll just check final logits
    
    # Summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    if logits_match:
        print("✅ PASS: HybridTPModel (tp=1) matches HuggingFace model!")
        print("   This indicates RoPE, weight loading, and forward pass are correct.")
    else:
        print("❌ FAIL: HybridTPModel (tp=1) does NOT match HuggingFace model!")
        print("   Likely issues:")
        print("   - RoPE shape/application mismatch")
        print("   - Weight loading/sharding error")
        print("   - Attention computation difference")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = test_golden_check()
        exit(0 if success else 1)
    finally:
        # Cleanup distributed if we initialized it
        if dist.is_initialized():
            dist.destroy_process_group()
