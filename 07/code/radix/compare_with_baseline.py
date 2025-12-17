"""
Compare RadixAttention output with baseline to verify correctness.
"""

import sys
import os
import subprocess

# Add paths
baseline_path = "/media/wukong/jackie/git.repo/distributed-ai/chapter6-distributed-inference-fundamentals-and-vllm/code/pa"
radix_path = "/media/wukong/jackie/git.repo/distributed-ai/chapter7-request-level-routing-and-sglang/code/radix"

def run_baseline():
    """Run baseline inference and capture output."""
    print("Running baseline inference...")
    result = subprocess.run(
        ["python", "inference_baseline.py"],
        cwd=baseline_path,
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout

def run_radix():
    """Run RadixAttention inference and capture output."""
    print("Running RadixAttention inference...")
    result = subprocess.run(
        ["python", "inference_radix.py"],
        cwd=radix_path,
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.stdout

def extract_generated_texts(output):
    """Extract generated text from output."""
    texts = []
    lines = output.split('\n')
    in_generated = False
    current_text = []
    
    for line in lines:
        if "Generated text:" in line:
            in_generated = True
            current_text = []
        elif in_generated:
            if line.strip() == "" and current_text:
                texts.append('\n'.join(current_text))
                current_text = []
                in_generated = False
            elif in_generated:
                current_text.append(line)
    
    if current_text:
        texts.append('\n'.join(current_text))
    
    return texts

def main():
    """Compare outputs."""
    print("=" * 60)
    print("Comparing RadixAttention with Baseline")
    print("=" * 60)
    
    baseline_output = run_baseline()
    radix_output = run_radix()
    
    baseline_texts = extract_generated_texts(baseline_output)
    radix_texts = extract_generated_texts(radix_output)
    
    print(f"\nBaseline generated {len(baseline_texts)} texts")
    print(f"RadixAttention generated {len(radix_texts)} texts")
    
    if len(baseline_texts) != len(radix_texts):
        print("WARNING: Different number of generated texts!")
        return
    
    all_match = True
    for i, (baseline_text, radix_text) in enumerate(zip(baseline_texts, radix_texts)):
        if baseline_text.strip() == radix_text.strip():
            print(f"\n✓ Prompt {i+1}: Outputs match!")
        else:
            print(f"\n✗ Prompt {i+1}: Outputs differ!")
            print(f"Baseline:\n{baseline_text[:200]}...")
            print(f"RadixAttention:\n{radix_text[:200]}...")
            all_match = False
    
    if all_match:
        print("\n" + "=" * 60)
        print("SUCCESS: All outputs match!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Some outputs differ!")
        print("=" * 60)

if __name__ == "__main__":
    main()
