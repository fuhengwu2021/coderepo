"""
Test script to demonstrate RadixAttention prefix sharing.

This script runs multiple requests with shared prefixes to show
how RadixAttention reuses cached KV values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import RadixAttentionModelWrapper
import time


def main():
    """Test prefix sharing with multiple requests."""
    print("=" * 60)
    print("RadixAttention Prefix Sharing Test")
    print("=" * 60)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Initialize model wrapper
    model_wrapper = RadixAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device
    )
    
    # Test prompts that share a common prefix (system prompt)
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",  # Shares prefix with first
        "What is the capital of Italy?",     # Shares prefix with first two
    ]
    
    print("Testing prefix sharing across multiple requests...")
    print("All prompts share the same system prompt prefix.\n")
    
    for i, prompt in enumerate(prompts):
        print(f"{'=' * 60}")
        print(f"Request {i + 1}: {prompt}")
        print(f"{'=' * 60}")
        
        start_time = time.time()
        generated = model_wrapper.generate(prompt, max_new_tokens=20)
        elapsed_time = time.time() - start_time
        
        # Extract just the assistant response
        if "assistant" in generated:
            assistant_part = generated.split("assistant")[-1].strip()
            print(f"\nResponse: {assistant_part[:100]}...")
        else:
            print(f"\nGenerated: {generated[:100]}...")
        
        print(f"Time: {elapsed_time:.2f}s\n")
    
    print("=" * 60)
    print("Prefix Sharing Summary:")
    print("=" * 60)
    print("The second and third requests should show 'Found X cached prefix tokens'")
    print("indicating that RadixAttention successfully reused cached KV values")
    print("from the shared system prompt prefix.")


if __name__ == "__main__":
    main()
