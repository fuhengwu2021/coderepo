#!/usr/bin/env python3
"""
Test script to verify both online softmax and safe_softmax algorithms work correctly.
"""

import os
import sys
# Add parent directory to path to import pa module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pa import PagedAttentionV2

def test_algorithm(use_online: bool, num_heads=2, head_dim=4, block_size=4, dtype=torch.float32):
    """Test a specific algorithm with simple synthetic data."""
    print(f"\n{'='*60}")
    print(f"Testing {'Online Softmax' if use_online else 'Safe Softmax'}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pa = PagedAttentionV2(
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_blocks=10,
        device=device,
        use_online_softmax=use_online
    )
    
    seq_id = 0
    
    # Create some synthetic K/V cache for 3 tokens
    num_tokens = 3
    for token_idx in range(num_tokens):
        k = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
        pa.append_kv(seq_id, k, v, token_idx)
    
    print(f"✓ Appended {num_tokens} tokens to sequence {seq_id}")
    
    # Create a query
    q = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
    
    # Compute attention
    output = pa.compute_attention(seq_id, q)
    
    print(f"✓ Computed attention output: shape {output.shape}, dtype={output.dtype}")
    print(f"  Output sample (first head): {output[0, :3].cpu().numpy()}")
    print(f"  Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    
    return output, pa

def compare_algorithms(dtype=torch.float32):
    """Compare outputs from both algorithms."""
    print(f"\n{'='*60}")
    print("Comparing Online Softmax vs Safe Softmax")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_heads = 2
    head_dim = 4
    block_size = 4
    
    # Initialize both algorithms
    pa_online = PagedAttentionV2(
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_blocks=10,
        device=device,
        use_online_softmax=True
    )
    
    pa_safe = PagedAttentionV2(
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_blocks=10,
        device=device,
        use_online_softmax=False
    )
    
    seq_id = 0
    num_tokens = 5
    
    # Use the same K/V cache for both
    kvs = []
    for token_idx in range(num_tokens):
        k = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
        kvs.append((k, v))
        pa_online.append_kv(seq_id, k, v, token_idx)
        pa_safe.append_kv(seq_id, k, v, token_idx)
    
    # Use the same query
    q = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
    
    # Compute attention with both algorithms
    output_online = pa_online.compute_attention(seq_id, q)
    output_safe = pa_safe.compute_attention(seq_id, q)
    
    # Compare results
    diff = torch.abs(output_online - output_safe)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nComparison Results:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  Relative error (max): {max_diff / (output_online.abs().max().item() + 1e-10):.8f}")
    
    if max_diff < 1e-5:
        print(f"  ✓ Results match closely (within numerical precision)")
    elif max_diff < 1e-3:
        print(f"  ⚠ Results are close but have some differences (likely numerical precision)")
    else:
        print(f"  ✗ Results differ significantly - possible bug!")
    
    return output_online, output_safe, max_diff

def test_with_real_model():
    """Test with a simple real model scenario."""
    print(f"\n{'='*60}")
    print("Testing with realistic model dimensions")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_heads = 14  # Qwen2.5-0.5B-Instruct
    head_dim = 64
    block_size = 16
    
    for use_online in [True, False]:
        algo_name = "Online Softmax" if use_online else "Safe Softmax"
        print(f"\n{algo_name}:")
        
        pa = PagedAttentionV2(
            block_size=block_size,
            num_heads=num_heads,
            head_dim=head_dim,
            max_blocks=100,
            device=device,
            use_online_softmax=use_online
        )
        
        seq_id = 0
        num_tokens = 20  # Simulate a short sequence
        
        # Append tokens
        for token_idx in range(num_tokens):
            k = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
            pa.append_kv(seq_id, k, v, token_idx)
        
        # Query
        q = torch.randn(num_heads, head_dim, device=device, dtype=torch.float16)
        
        # Compute attention
        output = pa.compute_attention(seq_id, q)
        
        print(f"  ✓ Processed {num_tokens} tokens")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Output dtype: {output.dtype}")
        print(f"  ✓ Output stats: mean={output.mean().item():.6f}, std={output.std().item():.6f}")

if __name__ == "__main__":
    print("="*60)
    print("Testing PagedAttention v2 - Both Algorithms")
    print("="*60)
    
    # Determine dtype based on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Test 1: Individual algorithm tests
    output_online, pa_online = test_algorithm(use_online=True, dtype=dtype)
    output_safe, pa_safe = test_algorithm(use_online=False, dtype=dtype)
    
    # Test 2: Compare both algorithms
    output_online_comp, output_safe_comp, max_diff = compare_algorithms(dtype=dtype)
    
    # Test 3: Test with realistic dimensions
    test_with_real_model()
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")
