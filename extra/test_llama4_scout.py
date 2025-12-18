#!/usr/bin/env python3
"""
Test script for Llama-4-Scout with vLLM or SGLang
Tests 2M context length (2097152 tokens) input + 200 tokens output
"""

import argparse
import requests
import json
import time
import sys
import random
from typing import Optional

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("⚠️  Warning: transformers not available, using approximate token counting")


def generate_prompt_text(input_length: int) -> str:
    """
    Generate prompt text with approximately input_length tokens.
    Uses the same logic for both vLLM and SGLang to ensure fair comparison.
    
    For small contexts (<100K): Uses tokenizer with sonnet.txt for accuracy.
    For large contexts (>=100K): Uses smart sampling from large_text_10mb.txt.
    """
    large_text_path = "/home/fuhwu/workspace/coderepo/extra/large_text_10mb.txt"
    sonnet_path = "/home/fuhwu/workspace/benchmark/genai-bench/genai_bench/data/sonnet.txt"
    
    # For large contexts (>100K), use fast approximation with conservative ratio
    # For smaller contexts, use tokenizer for accuracy
    if HAS_TOKENIZER and input_length < 100000:
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
            with open(sonnet_path, 'r') as f:
                base_text = f.read()
            
            prompt_text = base_text
            while len(tokenizer.encode(prompt_text, add_special_tokens=False)) < input_length:
                prompt_text += "\n\n" + base_text
            
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(tokens) > input_length:
                tokens = tokens[:input_length]
                prompt_text = tokenizer.decode(tokens)
            
            actual_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
            print(f"  ✅ Generated text with {actual_tokens:,} tokens (target: {input_length:,})")
            return prompt_text
        except Exception as e:
            print(f"  ⚠️  Tokenizer failed: {e}, using approximation")
            # Fallback to approximation
            try:
                with open(large_text_path, 'r') as f:
                    large_text = f.read()
                chars_per_token = 4.5  # Conservative estimate (ensures ≥2M, less overhead than 5.5) to ensure we reach target
                target_chars = int(input_length * chars_per_token)
                
                # Start at a random position to avoid prefix caching
                if len(large_text) >= target_chars:
                    max_start = len(large_text) - target_chars
                    start_pos = random.randint(0, max_start) if max_start > 0 else 0
                    prompt_text = large_text[start_pos:start_pos + target_chars]
                    print(f"  Using approximation from position {start_pos:,}: {len(prompt_text):,} characters ≈ {int(len(prompt_text) / chars_per_token):,} tokens")
                else:
                    num_repeats = (target_chars // len(large_text)) + 1
                    start_pos = random.randint(0, len(large_text) - 1) if len(large_text) > 0 else 0
                    prompt_text = large_text[start_pos:] + "\n"
                    remaining_chars = target_chars - len(prompt_text)
                    full_repeats_needed = remaining_chars // (len(large_text) + 1)
                    for _ in range(full_repeats_needed):
                        prompt_text += large_text + "\n"
                    remaining_chars = target_chars - len(prompt_text)
                    if remaining_chars > 0:
                        prompt_text += large_text[:remaining_chars]
                    prompt_text = prompt_text[:target_chars]
                    print(f"  Using approximation (repeated, starting at {start_pos:,}): {len(prompt_text):,} characters ≈ {int(len(prompt_text) / chars_per_token):,} tokens")
                return prompt_text
            except Exception as e2:
                print(f"  ⚠️  Fallback failed: {e2}")
                raise
    else:
        # For large contexts, use smart estimation: sample tokenizer to get actual ratio
        try:
            with open(large_text_path, 'r') as f:
                large_text = f.read()
            
            # Smart approach: sample tokenizer on a small portion to estimate actual ratio
            if HAS_TOKENIZER:
                try:
                    print(f"  Sampling tokenizer to estimate actual chars/token ratio...")
                    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
                    # Sample first 100K characters to estimate ratio
                    sample_size = min(100000, len(large_text))
                    sample_text = large_text[:sample_size]
                    sample_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                    actual_ratio = len(sample_text) / len(sample_tokens)
                    print(f"  Estimated ratio from sample: {actual_ratio:.2f} chars/token")
                    
                    # Use the sampled ratio directly (no buffer) to avoid exceeding limit
                    # The sample should be representative enough
                    chars_per_token = actual_ratio
                    print(f"  Using ratio: {chars_per_token:.2f} chars/token (from sample, no buffer to avoid exceeding 2M)")
                except Exception as e:
                    print(f"  ⚠️  Tokenizer sampling failed: {e}, using 4.1 (safe estimate)")
                    chars_per_token = 4.1  # Safe: 8.6M chars → ~2.08M tokens (slightly over but acceptable)
            else:
                # No tokenizer: use safe estimate
                chars_per_token = 4.1  # Safe: ensures ≥2M, won't exceed by much
                print(f"  No tokenizer available, using safe estimate: 4.1 chars/token")
            
            target_chars = int(input_length * chars_per_token)
            
            # Start at a random position to avoid prefix caching
            # This ensures fair performance comparison between different runs
            if len(large_text) >= target_chars:
                # Random starting position within the file
                max_start = len(large_text) - target_chars
                start_pos = random.randint(0, max_start) if max_start > 0 else 0
                prompt_text = large_text[start_pos:start_pos + target_chars]
                print(f"  Using large_text_10mb.txt (truncated from position {start_pos:,}): {len(prompt_text):,} characters")
            else:
                # Need to repeat the text, but start at random position for first chunk
                num_repeats = (target_chars // len(large_text)) + 1
                start_pos = random.randint(0, len(large_text) - 1) if len(large_text) > 0 else 0
                
                # Build text starting from random position, wrapping around
                prompt_text = large_text[start_pos:] + "\n"
                remaining_chars = target_chars - len(prompt_text)
                
                # Add full repeats
                full_repeats_needed = remaining_chars // (len(large_text) + 1)
                for _ in range(full_repeats_needed):
                    prompt_text += large_text + "\n"
                
                # Add final partial chunk from beginning
                remaining_chars = target_chars - len(prompt_text)
                if remaining_chars > 0:
                    prompt_text += large_text[:remaining_chars]
                
                # Trim to exact target
                prompt_text = prompt_text[:target_chars]
                print(f"  Using large_text_10mb.txt (repeated {num_repeats}x, starting at position {start_pos:,}): {len(prompt_text):,} characters")
            print(f"  Estimated tokens: {int(len(prompt_text) / chars_per_token):,} (target: {input_length:,})")
            return prompt_text
        except Exception as e:
            print(f"  ❌ Failed to read large_text_10mb.txt: {e}")
            raise


def test_vllm(
    base_url: str,
    input_length: int,
    output_length: int,
    model_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
):
    """Test vLLM with specified context length"""
    print(f"Testing vLLM with {input_length} input tokens + {output_length} output tokens")
    
    # Generate prompt text using shared function (same as SGLang for fair comparison)
    prompt_text = generate_prompt_text(input_length)
    
    payload = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "max_tokens": output_length,
        "temperature": 0.7,
    }
    
    print(f"Sending request to {base_url}/v1/chat/completions")
    print(f"Input length (approx): {len(prompt_text)} characters")
    print(f"Expected output tokens: {output_length}")
    print()
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=3600  # 1 hour timeout for large context
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            output_text = result["choices"][0]["message"]["content"]
            tokens_used = result.get("usage", {})
            
            print("✅ Success!")
            print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
            print(f"📊 Tokens used: {tokens_used}")
            print(f"📝 Output length: {len(output_text)} characters")
            print(f"📝 Output preview: {output_text[:200]}...")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_sglang(
    base_url: str,
    input_length: int,
    output_length: int,
    model_path: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
):
    """Test SGLang with specified context length"""
    print(f"Testing SGLang with {input_length} input tokens + {output_length} output tokens")
    
    # Generate prompt text using shared function (same as vLLM for fair comparison)
    prompt_text = generate_prompt_text(input_length)
    
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": output_length,
            "temperature": 0.7,
        }
    }
    
    print(f"Sending request to {base_url}/generate")
    print(f"Input length (approx): {len(prompt_text)} characters")
    print(f"Expected output tokens: {output_length}")
    print()
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            timeout=3600  # 1 hour timeout for large context
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            output_text = result.get("text", "")
            
            print("✅ Success!")
            print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds")
            print(f"📝 Output length: {len(output_text)} characters")
            print(f"📝 Output preview: {output_text[:200]}...")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Error: Request timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def check_health(base_url: str) -> bool:
    """Check if the service is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"⚠️  Health check failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Llama-4-Scout with vLLM or SGLang"
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "sglang"],
        required=True,
        help="Backend to test (vllm or sglang)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the service (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=2097152,
        help="Input context length in tokens (default: 2097152 = 2M)"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=200,
        help="Output length in tokens (default: 200)"
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip health check before testing"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Testing Llama-4-Scout with {args.backend.upper()}")
    print("=" * 60)
    print()
    
    # Health check
    if not args.skip_health_check:
        print("🔍 Checking service health...")
        if not check_health(args.url):
            print("❌ Service is not healthy. Please check the deployment.")
            sys.exit(1)
        print("✅ Service is healthy")
        print()
    
    # Run test
    if args.backend == "vllm":
        success = test_vllm(args.url, args.input_length, args.output_length)
    else:
        success = test_sglang(args.url, args.input_length, args.output_length)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
