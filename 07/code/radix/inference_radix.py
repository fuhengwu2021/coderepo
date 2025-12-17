"""
RadixAttention inference script - main entry point.

This script runs inference using RadixAttention and compares with baseline.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import RadixAttentionModelWrapper
import time


def main():
    """Main function to run RadixAttention inference."""
    print("=" * 60)
    print("RadixAttention Inference (Prefix Cache Reuse)")
    print("=" * 60)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = RadixAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device
    )
    
    # Test prompts (same as baseline)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {prompt}")
        print(f"{'=' * 60}")
        
        start_time = time.time()
        generated = model_wrapper.generate(prompt, max_new_tokens=50)
        elapsed_time = time.time() - start_time
        
        print(f"\nGenerated text:")
        print(generated)
        print(f"\nTime taken: {elapsed_time:.2f} seconds")
        print()
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("Final Memory Stats:")
    print(f"{'=' * 60}")
    stats = model_wrapper.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
