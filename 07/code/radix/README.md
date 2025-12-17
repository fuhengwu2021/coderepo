# RadixAttention Implementation

This directory contains a simplified implementation of SGLang's RadixAttention technique for prefix cache reuse.

## Overview

RadixAttention uses a radix tree (prefix tree) data structure to store and share KV cache across requests that share common prefixes. This is particularly valuable when multiple requests share the same prompt prefix, such as AI agents with system prompts.

## Key Components

- **`radix_cache.py`**: Implements the radix tree data structure (`RadixKey`, `TreeNode`, `RadixCache`) for storing and matching KV cache prefixes
- **`radix_attention.py`**: Implements the RadixAttention layer that uses RadixCache for prefix cache reuse
- **`inference.py`**: Main model wrapper that integrates RadixAttention with HuggingFace models
- **`inference_radix.py`**: Entry point script for running inference

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python inference_radix.py
```

## Features

1. **Prefix Matching**: Automatically detects when new requests share prefixes with cached requests
2. **KV Cache Reuse**: Reuses cached KV values for shared prefixes, avoiding redundant computation
3. **Radix Tree Storage**: Efficiently stores and retrieves cached prefixes using a radix tree structure

## Comparison with Baseline

The implementation produces the same results as the baseline (`chapter6-distributed-inference-fundamentals-and-vllm/code/pa/inference_baseline.py`) while demonstrating RadixAttention's prefix cache reuse capability.

When running multiple prompts that share a common prefix (e.g., system prompt), you'll see messages like:
```
[Prefill] Sequence 1: Found 24 cached prefix tokens
```

This indicates that the radix cache successfully matched and reused 24 tokens from a previous request.

## Architecture

The implementation follows SGLang's approach:

1. **RadixCache**: Maintains a radix tree of cached KV prefixes
2. **Prefix Matching**: When a new request arrives, `match_prefix()` finds the longest matching prefix
3. **KV Reuse**: Cached KV values are reused, and only new tokens are computed
4. **Cache Insertion**: New prefixes are inserted into the radix tree for future reuse

## Limitations

This is a simplified educational implementation. The full SGLang implementation includes:
- More sophisticated eviction policies
- Page-level caching (not just token-level)
- Optimized CUDA kernels
- Support for more complex scenarios (LoRA, multi-GPU, etc.)
