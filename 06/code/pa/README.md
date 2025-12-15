# Simple PagedAttention Implementation

This directory contains a minimal but functional implementation of PagedAttention for managing KV cache in LLM inference. It includes three versions:

- **v1** (`paged_attention.py`): Basic implementation with block concatenation
- **v2** (`paged_attention_v2.py`): Advanced implementation with online softmax and GQA support
- **v3** (`paged_attention_v3.py`): Continuous batching support (sequential processing with scheduler)
- **v4** (`paged_attention_v4.py`): Ragged batching metadata structure (demonstrates flattened tokens + metadata)

## Overview

PagedAttention solves two critical problems in LLM serving:

1. **KV Cache Fragmentation**: By using fixed-size blocks (similar to OS pages), it eliminates memory fragmentation when serving multiple requests with different sequence lengths.

2. **Padding FLOPs Elimination**: Instead of padding sequences to max length, attention computation iterates only over blocks that actually exist for each sequence, eliminating wasted computation on padding tokens.

## Architecture

### Core Components

- **`block_manager.py`**: Manages allocation and deallocation of fixed-size blocks
  - `Block`: A fixed-size container for KV cache (e.g., 16 tokens)
  - `BlockTable`: Maps logical token positions to physical blocks for each sequence
  - `BlockManager`: Allocator that maintains a pool of blocks and reuses freed blocks

- **`paged_attention.py`**: Basic block-based attention computation (v1)
  - `PagedAttention`: Computes attention by concatenating blocks, then standard attention

- **`paged_attention_v2.py`**: Advanced block-streaming attention computation (v2)
  - `PagedAttentionV2`: Computes attention using online softmax (single-pass) or safe_softmax (two-pass)
  - Supports GQA via reshape + broadcast (no physical KV repeat)
  - More memory efficient and avoids materializing large concatenated tensors

- **`paged_attention_v3.py`**: Continuous batching support (v3)
  - `PagedAttentionV3`: Extends v2 with batch attention computation
  - Supports multiple sequences processed concurrently
  - Uses scheduler for dynamic sequence management

- **`paged_attention_v4.py`**: Ragged batching metadata structure (v4)
  - `PagedAttentionV4`: Demonstrates flattened tokens + metadata arrays
  - Implements prefill batching with metadata
  - Shows ragged batching structure (though processing is still sequential due to HuggingFace limitations)

- **`scheduler.py`**: Continuous batching scheduler
  - `ContinuousBatchScheduler`: Manages sequence states (prefill, decode, finished)
  - `SequenceInfo`: Tracks sequence state and tokens
  - `get_prefill_batch()`: Returns sequences ready for prefill batching

- **`inference.py`**: Example inference script using PagedAttention v1
- **`inference_v2.py`**: Example inference script using PagedAttention v2 with online/safe softmax
- **`inference_v3.py`**: Example inference script using PagedAttention v3 with continuous batching
- **`inference_v4.py`**: Example inference script using PagedAttention v4 with ragged batching metadata

## Key Features

### Version 1 (Basic)
1. **Block-based Storage**: KV cache is stored in fixed-size blocks (default: 16 tokens)
2. **No Padding**: Attention only computes over tokens that exist, not padded positions
3. **Dynamic Allocation**: Blocks are allocated on-demand and freed when sequences complete
4. **Block Reuse**: Freed blocks are returned to a pool for reuse, reducing fragmentation

### Version 2 (Advanced)
All features from v1, plus:
1. **Online Softmax**: Single-pass algorithm with running max + rescale (more efficient)
2. **Safe Softmax**: Two-pass algorithm (first pass finds global max, second computes weighted sum)
3. **GQA Support**: Efficient GQA via reshape + broadcast (no physical KV repeat, saves memory)
4. **Block Streaming**: Processes blocks one-by-one without concatenation

### Version 3 (Continuous Batching - Pseudo Implementation)
All features from v2, plus:
1. **Sequence Scheduling**: Scheduler manages multiple sequences and their states
2. **Shared Block Pool**: Multiple sequences share blocks (no padding needed)
3. **Dynamic Management**: Sequences can be added/removed dynamically
4. **State Tracking**: Proper prefill → decode → finished transitions

**Note**: This is NOT true ragged batching. We use sequential processing (Python loops)
rather than flattened tokens + metadata.

### Version 4 (Ragged Batching Metadata Structure)
All features from v2, plus:
1. **Ragged Batching Metadata**: Flattened tokens + metadata arrays (seq_id, position, slot_mapping)
2. **Prefill Batching**: Multiple prompts processed together with metadata
3. **Metadata Structure**: Demonstrates how to organize tokens for ragged batching
4. **No Padding**: T tokens instead of B×Lmax (demonstrates the concept)

**Note**: This demonstrates the **metadata structure** for ragged batching, but actual processing
is still sequential due to HuggingFace limitations. True ragged batching requires custom CUDA
kernels that process flattened tokens `[T, D]` with metadata in a single kernel call.

## Usage

### Basic Example (v1)

```python
from pa import PagedAttention

# Initialize PagedAttention
pa = PagedAttention(
    block_size=16,
    num_heads=32,
    head_dim=128,
    device="cuda"
)

# Append KV cache for tokens
for token_idx in range(seq_len):
    k = ...  # [num_heads, head_dim]
    v = ...  # [num_heads, head_dim]
    pa.append_kv(seq_id=0, k=k, v=v, token_idx=token_idx)

# Compute attention (only over existing blocks, no padding)
q = ...  # [num_heads, head_dim]
output = pa.compute_attention(seq_id=0, q=q)
```

### Advanced Example (v2 with GQA)

```python
from pa import PagedAttentionV2

# Initialize PagedAttention v2 with GQA support
pa = PagedAttentionV2(
    block_size=16,
    num_heads=14,        # Q heads (Hq)
    head_dim=64,
    device="cuda",
    use_online_softmax=True,  # Use online softmax (default) or False for safe_softmax
    num_kv_heads=2       # KV heads (Hkv) for GQA
)

# Append KV cache (with num_kv_heads, not num_heads)
for token_idx in range(seq_len):
    k = ...  # [num_kv_heads, head_dim] - no physical repeat!
    v = ...  # [num_kv_heads, head_dim] - no physical repeat!
    pa.append_kv(seq_id=0, k=k, v=v, token_idx=token_idx)

# Compute attention (GQA handled via reshape + broadcast)
q = ...  # [num_heads, head_dim]
output = pa.compute_attention(seq_id=0, q=q)  # Returns [num_heads, head_dim]
```

### Running Inference

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference with PagedAttention v1
python inference.py

# Run inference with PagedAttention v2 (online softmax)
python inference_v2.py

# Run inference with PagedAttention v3 (continuous batching)
python inference_v3.py

# Run inference with PagedAttention v4 (ragged batching metadata)
python inference_v4.py

# Run baseline inference (traditional KV cache) for comparison
python inference_baseline.py
```

All inference scripts will:
1. Load the Qwen/Qwen2.5-0.5B-Instruct model
2. Process two prompts using the respective KV cache management approach
3. Apply RoPE using HuggingFace's native implementation
4. Generate text while demonstrating memory/block usage
5. Print timing information and statistics

**Note**: The implementation is tested with Qwen2.5 models. For other model families, you may need to adjust the RoPE access pattern (some models store `rotary_emb` on individual attention layers rather than at the model level).

## How It Works

### Traditional Approach (with Padding)

```
Batch of 4 sequences with lengths [128, 64, 32, 8]:
- All padded to max_len = 128
- Attention computes over 128 tokens for ALL sequences
- Padding FLOPs: ~67% wasted computation
```

### PagedAttention Approach

```
Same batch:
- Sequence 1: 8 blocks (128 tokens)
- Sequence 2: 4 blocks (64 tokens)  
- Sequence 3: 2 blocks (32 tokens)
- Sequence 4: 1 block (8 tokens)

Attention iterates only over blocks that exist:
- Sequence 4 only computes over 1 block (8 tokens)
- No padding tokens are ever computed
- Padding FLOPs = 0
```

## Architecture and Data Structures

### Data Structure Hierarchy

```
BlockManager (per layer)
│
├── free_blocks: List[Block]          # Pool of available blocks
├── allocated_blocks: dict[block_id → Block]
└── sequence_tables: dict[seq_id → BlockTable]
    │
    └── BlockTable (per sequence)
        └── blocks: List[Block]       # Ordered blocks for this sequence
            │
            └── Block
                ├── block_id: int
                ├── k_cache: [block_size, num_kv_heads, head_dim]  # v2: uses num_kv_heads
                ├── v_cache: [block_size, num_kv_heads, head_dim]  # v2: uses num_kv_heads
                └── num_tokens: int   # Valid tokens (0 ≤ num_tokens ≤ block_size)
```

### Memory Layout Example

```
Sequence 0 (36 tokens, block_size=16):
┌─────────────────────────────────────────────────────────────┐
│ BlockTable (seq_id=0)                                       │
│                                                              │
│  Block[0] (block_id=5)        Block[1] (block_id=12)        │
│  ┌──────────────────┐        ┌──────────────────┐         │
│  │ Tokens 0-15      │        │ Tokens 16-31     │         │
│  │ k_cache[16,Hkv,D]│        │ k_cache[16,Hkv,D]│         │
│  │ v_cache[16,Hkv,D]│        │ v_cache[16,Hkv,D]│         │
│  │ num_tokens=16    │        │ num_tokens=16    │         │
│  └──────────────────┘        └──────────────────┘         │
│                                                              │
│  Block[2] (block_id=23)                                     │
│  ┌──────────────────┐                                      │
│  │ Tokens 32-35     │                                      │
│  │ k_cache[16,Hkv,D]│                                      │
│  │ v_cache[16,Hkv,D]│                                      │
│  │ num_tokens=4     │  ← Partially filled                 │
│  └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘

Note: For v2 with GQA, blocks store KV with num_kv_heads (not num_heads),
      saving memory. GQA mapping is done via reshape + broadcast during
      attention computation.
```

## Implementation Details

### Version 1: Basic PagedAttention

**Attention Computation**:
1. Collect K/V from all blocks
2. Concatenate into single tensor
3. Compute standard attention (Q @ K^T, softmax, @ V)

**Pros**: Simple, easy to understand
**Cons**: Requires materializing concatenated tensor (O(L) memory)

### Version 2: Advanced PagedAttention

**Online Softmax Algorithm** (single-pass):
1. Initialize running_max, weighted_sum, sum_exp
2. For each block:
   - Compute scores = Q @ K^T * scale
   - Update running_max if block_max > running_max
   - Rescale previous accumulations if max increased
   - Accumulate exp(scores - running_max) * V
3. Normalize: output = weighted_sum / sum_exp

**Safe Softmax Algorithm** (two-pass):
1. **Pass 1**: Find global_max across all blocks
2. **Pass 2**: Compute exp(scores - global_max) * V and accumulate

**GQA Implementation** (reshape + broadcast):
- Q: `[num_heads, head_dim]` → reshape to `[num_kv_heads, g, head_dim]` where `g = num_heads // num_kv_heads`
- K/V: Stored as `[num_kv_heads, head_dim]` (no physical repeat)
- Attention: Use einsum `'hgd,hkd->hgk'` to compute scores, then `'hgk,hkd->hgd'` for output
- Output: Reshape back to `[num_heads, head_dim]`

**Advantages**:
- No physical KV repeat (saves memory)
- No large concatenated tensors (block streaming)
- More efficient (online softmax: single pass, one matmul per block)

### RoPE (Rotary Position Embedding) Integration

This implementation uses HuggingFace's native RoPE implementation from Qwen2 models to ensure correctness:

- **Shared RoPE Module**: The rotary embedding is accessed via `model.model.rotary_emb` (shared across all layers)
- **Native HF Functions**: Uses `apply_rotary_pos_emb` from `transformers.models.qwen2.modeling_qwen2`
- **GQA Support**: v2 handles GQA efficiently via reshape + broadcast (no physical KV repeat)

The implementation follows the same pattern as Qwen2Attention:
1. Get cos/sin from `model.model.rotary_emb(hidden_states, position_ids)`
2. Apply rotation using `apply_rotary_pos_emb(query_states, key_states, cos, sin)`

### Prefill vs Decode

- **Prefill**: Uses HuggingFace's `model.forward(use_cache=True)` to get initial KV cache with RoPE applied, then extracts and stores in PagedAttention blocks
- **Decode**: Manually processes each layer, applies RoPE using HF's implementation, then uses `PagedAttention.compute_attention()` (v1) or `PagedAttentionV2.compute_attention()` (v2) for the actual attention computation

## Code Execution Flow

### Prefill Phase

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Input: prompt text                                        │
│    ↓                                                          │
│ 2. Tokenize & apply chat template                            │
│    ↓                                                          │
│ 3. model.forward(input_ids, use_cache=True)                  │
│    ├── Applies RoPE automatically                            │
│    ├── Computes attention for all tokens                     │
│    └── Returns past_key_values (KV cache)                    │
│    ↓                                                          │
│ 4. Extract KV from past_key_values[layer_idx]                │
│    k: [batch, num_kv_heads, seq_len, head_dim]              │
│    v: [batch, num_kv_heads, seq_len, head_dim]              │
│    ↓                                                          │
│ 5. For each token in sequence:                               │
│    ├── Get k_token: [num_kv_heads, head_dim]                 │
│    ├── Get v_token: [num_kv_heads, head_dim]                │
│    └── pa.append_kv(seq_id, k_token, v_token, token_idx)    │
│        ├── Allocate block if needed                          │
│        ├── Store k_token in block.k_cache[slot]             │
│        └── Store v_token in block.v_cache[slot]              │
│        Note: v2 stores with num_kv_heads (no repeat)         │
│    ↓                                                          │
│ 6. KV cache now stored in PagedAttention blocks             │
└──────────────────────────────────────────────────────────────┘
```

### Decode Phase (v2 with Online Softmax)

```
┌──────────────────────────────────────────────────────────────┐
│ For each layer (24 layers for Qwen2.5-0.5B):                │
│                                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Layer Processing:                                        │ │
│ │                                                           │ │
│ │ 1. Embed current token                                   │ │
│ │    hidden_states = embed_tokens([token_id])              │ │
│ │                                                           │ │
│ │ 2. Apply input layer norm                                 │ │
│ │    hidden_states_norm = input_layernorm(hidden_states)   │ │
│ │                                                           │ │
│ │ 3. Compute Q, K, V projections                            │ │
│ │    q_proj = q_proj(hidden_states_norm)                  │ │
│ │    k_proj = k_proj(hidden_states_norm)                  │ │
│ │    v_proj = v_proj(hidden_states_norm)                  │ │
│ │                                                           │ │
│ │ 4. Reshape to [B, H, seq_len, D]                        │ │
│ │    q: [1, num_heads, 1, head_dim]                       │ │
│ │    k: [1, num_kv_heads, 1, head_dim]                    │ │
│ │    v: [1, num_kv_heads, 1, head_dim]                    │ │
│ │                                                           │ │
│ │ 5. Apply RoPE (using HF's implementation)                │ │
│ │    cos, sin = model.model.rotary_emb(k, position_ids)   │ │
│ │    q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)│ │
│ │                                                           │ │
│ │ 6. PagedAttentionV2.compute_attention(seq_id, q_rope)   │ │
│ │    ├── Reshape Q: [Hq, D] → [Hkv, g, D]                 │ │
│ │    ├── For each block (single pass):                     │ │
│ │    │   ├── scores = einsum('hgd,hkd->hgk') * scale      │ │
│ │    │   ├── Update running_max if needed                  │ │
│ │    │   ├── Rescale previous accumulations                │ │
│ │    │   ├── Accumulate exp(scores - max) * V              │ │
│ │    │   └── Accumulate sum_exp                            │ │
│ │    ├── Normalize: output = weighted_sum / sum_exp        │ │
│ │    └── Reshape output: [Hkv, g, D] → [Hq, D]             │ │
│ │                                                           │ │
│ │ 7. Cache new K/V for current token                       │ │
│ │    pa.append_kv(seq_id, k_rope, v, token_idx)           │ │
│ │    Note: Stores with num_kv_heads (no repeat)             │ │
│ │                                                           │ │
│ │ 8. Output projection & residual                          │ │
│ │    attn_output = o_proj(attn_output)                     │ │
│ │    hidden_states = hidden_states + attn_output           │ │
│ │                                                           │ │
│ │ 9. MLP & residual                                         │ │
│ │    mlp_output = mlp(post_attention_layernorm(...))      │ │
│ │    hidden_states = hidden_states + mlp_output            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                               │
│ 10. Final layer norm & LM head                               │
│     logits = lm_head(norm(hidden_states))                    │
│     ↓                                                         │
│ 11. Sample next token                                        │
└──────────────────────────────────────────────────────────────┘
```

### Attention Computation Details

#### Version 1: Basic PagedAttention

```
PagedAttention.compute_attention(seq_id, q):
┌──────────────────────────────────────────────────────────────┐
│ Input:                                                        │
│   q: [num_heads, head_dim]                                   │
│   seq_id: sequence identifier                                │
│                                                               │
│ Step 1: Get BlockTable                                        │
│   block_table = block_manager.get_block_table(seq_id)        │
│                                                               │
│ Step 2: Collect K/V from blocks                              │
│   k_list = []                                                 │
│   v_list = []                                                 │
│   for block in block_table.blocks:                           │
│       num_valid = block.num_tokens  # e.g., 16, 16, 4       │
│       k_list.append(block.k_cache[:num_valid])               │
│       v_list.append(block.v_cache[:num_valid])               │
│                                                               │
│ Step 3: Concatenate                                           │
│   k_cached = cat(k_list)  # [total_tokens, num_heads, D]     │
│   v_cached = cat(v_list)  # [total_tokens, num_heads, D]     │
│                                                               │
│ Step 4: Compute attention scores                             │
│   q_expanded = q.unsqueeze(1)      # [H, 1, D]               │
│   k_transposed = k_cached.transpose(0,1)  # [H, T, D]        │
│   scores = (q_expanded @ k_transposed^T) * scale              │
│            # [H, 1, D] @ [H, D, T] = [H, 1, T]              │
│                                                               │
│ Step 5: Softmax                                               │
│   attn_weights = softmax(scores, dim=-1)  # [H, T]           │
│                                                               │
│ Step 6: Weighted sum of values                                │
│   v_transposed = v_cached.transpose(0,1)  # [H, T, D]        │
│   output = attn_weights @ v_transposed    # [H, T] @ [H,T,D] │
│            # [H, 1, T] @ [H, T, D] = [H, 1, D]              │
│                                                               │
│ Output: [num_heads, head_dim]                                │
└──────────────────────────────────────────────────────────────┘
```

#### Version 2: Advanced PagedAttention (Online Softmax)

```
PagedAttentionV2.compute_attention(seq_id, q) [Online Softmax]:
┌──────────────────────────────────────────────────────────────┐
│ Input:                                                        │
│   q: [num_heads, head_dim]                                   │
│   seq_id: sequence identifier                                │
│                                                               │
│ Step 1: Reshape Q for GQA                                     │
│   q_reshaped = q.view(num_kv_heads, g, head_dim)            │
│   # [Hq, D] → [Hkv, g, D] where g = Hq // Hkv               │
│                                                               │
│ Step 2: Initialize running statistics                        │
│   running_max = [-inf, ..., -inf]  # [Hkv, g]                │
│   weighted_sum = zeros([Hkv, g, D])                          │
│   sum_exp = zeros([Hkv, g])                                  │
│                                                               │
│ Step 3: Single pass over blocks                              │
│   for block in block_table.blocks:                           │
│       num_valid = block.num_tokens                           │
│       k_block = block.k_cache[:num_valid]  # [T, Hkv, D]     │
│       v_block = block.v_cache[:num_valid]  # [T, Hkv, D]     │
│                                                               │
│       # Transpose for computation                            │
│       k_block_t = k_block.transpose(0, 1)  # [Hkv, T, D]     │
│       v_block_t = v_block.transpose(0, 1)  # [Hkv, T, D]     │
│                                                               │
│       # Compute scores: [Hkv, g, D] @ [Hkv, D, T]            │
│       scores = einsum('hgd,hkd->hgk', q_reshaped, k_block_t) │
│                * scale  # [Hkv, g, T]                        │
│                                                               │
│       # Find max for this block                              │
│       block_max = max(scores, dim=-1)  # [Hkv, g]            │
│                                                               │
│       # Rescale if max increased                             │
│       should_rescale = block_max > running_max               │
│       new_max = max(running_max, block_max)                  │
│       rescale_factor = where(should_rescale,                 │
│                              exp(running_max - new_max), 1)  │
│       weighted_sum *= rescale_factor.unsqueeze(-1)           │
│       sum_exp *= rescale_factor                               │
│       running_max = new_max                                   │
│                                                               │
│       # Compute exp(scores - running_max)                     │
│       exp_scores = exp(scores - running_max.unsqueeze(-1))   │
│                                                               │
│       # Accumulate                                            │
│       sum_exp += sum(exp_scores, dim=-1)  # [Hkv, g]         │
│       weighted_sum += einsum('hgk,hkd->hgd',                  │
│                              exp_scores, v_block_t)          │
│                                                               │
│ Step 4: Normalize                                             │
│   output_reshaped = weighted_sum / sum_exp.unsqueeze(-1)     │
│   # [Hkv, g, D]                                               │
│                                                               │
│ Step 5: Reshape back                                          │
│   output = output_reshaped.view(num_heads, head_dim)         │
│   # [Hkv, g, D] → [Hq, D]                                    │
│                                                               │
│ Output: [num_heads, head_dim]                                │
└──────────────────────────────────────────────────────────────┘
```

#### Version 2: Advanced PagedAttention (Safe Softmax)

```
PagedAttentionV2.compute_attention(seq_id, q) [Safe Softmax]:
┌──────────────────────────────────────────────────────────────┐
│ Input:                                                        │
│   q: [num_heads, head_dim]                                   │
│   seq_id: sequence identifier                                │
│                                                               │
│ Step 1: Reshape Q for GQA                                     │
│   q_reshaped = q.view(num_kv_heads, g, head_dim)            │
│   # [Hq, D] → [Hkv, g, D]                                    │
│                                                               │
│ PASS 1: Find global maximum                                  │
│   global_max = [-inf, ..., -inf]  # [Hkv, g]                 │
│   for block in block_table.blocks:                           │
│       scores = einsum('hgd,hkd->hgk', q_reshaped, k_block_t)│
│                * scale  # [Hkv, g, T]                        │
│       block_max = max(scores, dim=-1)  # [Hkv, g]            │
│       global_max = max(global_max, block_max)                │
│                                                               │
│ PASS 2: Compute weighted sum                                 │
│   weighted_sum = zeros([Hkv, g, D])                          │
│   sum_exp = zeros([Hkv, g])                                  │
│   for block in block_table.blocks:                           │
│       scores = einsum('hgd,hkd->hgk', q_reshaped, k_block_t)│
│                * scale  # [Hkv, g, T]                        │
│       exp_scores = exp(scores - global_max.unsqueeze(-1))    │
│       sum_exp += sum(exp_scores, dim=-1)                     │
│       weighted_sum += einsum('hgk,hkd->hgd',                 │
│                              exp_scores, v_block_t)          │
│                                                               │
│ Step 3: Normalize and reshape                                │
│   output_reshaped = weighted_sum / sum_exp.unsqueeze(-1)     │
│   output = output_reshaped.view(num_heads, head_dim)         │
│                                                               │
│ Output: [num_heads, head_dim]                                │
└──────────────────────────────────────────────────────────────┘
```

### Block Allocation Flow

```
append_kv(seq_id, k, v, token_idx):
┌──────────────────────────────────────────────────────────────┐
│ 1. Get or create BlockTable for seq_id                       │
│    block_table = get_block_table(seq_id)                     │
│                                                               │
│ 2. Check if we need a new block:                             │
│    last_block = block_table.get_last_block()                 │
│    if last_block is None or last_block.is_full():            │
│        ┌───────────────────────────────────────────────────┐ │
│        │ Allocate new block:                              │ │
│        │   if free_blocks is empty:                        │ │
│        │       allocate new Block from memory              │ │
│        │   else:                                           │ │
│        │       block = free_blocks.pop()  # Reuse!          │ │
│        │   block_table.append_block(block)                 │ │
│        └───────────────────────────────────────────────────┘ │
│                                                               │
│ 3. Store K/V in the block:                                    │
│    slot = last_block.num_tokens                              │
│    last_block.k_cache[slot] = k  # [num_kv_heads, head_dim]  │
│    last_block.v_cache[slot] = v  # [num_kv_heads, head_dim]  │
│    Note: v1 uses num_heads, v2 uses num_kv_heads            │
│    last_block.num_tokens += 1                                │
│                                                               │
│ 4. Block is now updated with new token                       │
└──────────────────────────────────────────────────────────────┘
```

### Block Deallocation Flow

Blocks are released (returned to the free pool) when a sequence completes:

```
free_sequence(seq_id):
┌──────────────────────────────────────────────────────────────┐
│ 1. Check if sequence exists                                   │
│    if seq_id not in sequence_tables:                          │
│        return  # Nothing to free                              │
│                                                               │
│ 2. Get BlockTable for the sequence                           │
│    block_table = sequence_tables[seq_id]                     │
│                                                               │
│ 3. Return all blocks to free pool                            │
│    for block in block_table.blocks:                          │
│        block.num_tokens = 0  # Reset block state             │
│        free_blocks.append(block)  # Return to pool           │
│                                                               │
│ 4. Remove sequence table                                      │
│    del sequence_tables[seq_id]                               │
│                                                               │
│ Result: All blocks from this sequence are now available      │
│         for reuse by future sequences                        │
└──────────────────────────────────────────────────────────────┘
```

**When blocks are released:**

1. **After generation completes**: When `generate()` finishes (either reaching `max_new_tokens`, encountering EOS token, or error), it calls `free_sequence(seq_id)` for all layers
2. **Manual cleanup**: You can explicitly call `free_sequence(seq_id)` to release blocks before generation completes
3. **Block reuse**: Freed blocks are added back to `free_blocks` pool and can be immediately reused by new sequences, reducing memory fragmentation

**Example lifecycle:**

```
Sequence 0: Allocates blocks [5, 12, 23]
  ↓ (generation completes)
free_sequence(0): Returns blocks [5, 12, 23] to free_blocks
  ↓
Sequence 1: Reuses block 5 (from free_blocks.pop())
  ↓ (generation completes)
free_sequence(1): Returns block 5 back to free_blocks
```

This block reuse mechanism is key to PagedAttention's efficiency: it eliminates memory fragmentation by reusing fixed-size blocks across different sequences.

## Implementation Notes

This is a **simplified educational implementation** to demonstrate the core concepts. Production systems like vLLM have additional optimizations:

1. **Kernel-level Integration**: Attention kernels are custom-written to work directly with block tables
2. **Continuous Batching**: Blocks enable efficient dynamic batching
3. **Prefix Caching**: Shared blocks for common prefixes
4. **Multi-GPU Support**: Distributed block management
5. **Custom CUDA Kernels**: Highly optimized attention kernels for block-based computation

## Files

- **`block_manager.py`**: Core block allocation and management
- **`paged_attention.py`**: Basic block-based attention computation (v1)
- **`paged_attention_v2.py`**: Advanced block-streaming attention with online softmax and GQA support (v2)
- **`paged_attention_v3.py`**: Continuous batching support with scheduler (v3)
- **`scheduler.py`**: Continuous batching scheduler for managing multiple sequences
- **`inference.py`**: PagedAttention v1-based inference with HuggingFace RoPE integration
- **`inference_v2.py`**: PagedAttention v2-based inference with online/safe softmax options
- **`inference_v3.py`**: PagedAttention v3-based inference with continuous batching
- **`inference_baseline.py`**: Baseline inference for comparison (traditional KV cache)
- **`requirements.txt`**: Python dependencies (torch, transformers, accelerate)

## Performance Comparison

The implementation includes timing measurements. Typical results on Qwen2.5-0.5B-Instruct:

- **Baseline**: Traditional KV cache with padding
- **v1**: Basic PagedAttention (concatenates blocks)
- **v2 (Online Softmax)**: Single-pass block streaming (most efficient)
- **v2 (Safe Softmax)**: Two-pass block streaming (more straightforward)
- **v3 (Continuous Batching)**: Multiple sequences processed concurrently (highest throughput)

All implementations correctly handle GQA, with v2/v3 being more memory efficient (no physical KV repeat).

## Version Comparison

| Feature | v1 | v2 | v3 | v4 |
|---------|----|----|----|----|
| Block Management | ✅ | ✅ | ✅ | ✅ |
| Online Softmax | ❌ | ✅ | ✅ | ✅ |
| GQA Support | ❌ | ✅ | ✅ | ✅ |
| Sequence Scheduler | ❌ | ❌ | ✅ | ✅ |
| Shared Block Pool | ❌ | ❌ | ✅ | ✅ |
| Ragged Metadata Structure | ❌ | ❌ | ❌ | ✅ |
| Prefill Batching | ❌ | ❌ | ❌ | ✅ |
| True Ragged Processing | ❌ | ❌ | ❌ | ❌ (HuggingFace limitation) |
| Parallel Batch Processing | ❌ | ❌ | ❌ (sequential) | ❌ (sequential) |

**v3 Note**: Provides scheduling and shared blocks, but processes sequences sequentially.

**v4 Note**: Demonstrates ragged batching metadata structure (flattened tokens + metadata arrays),
but processing is still sequential due to HuggingFace limitations. True ragged batching requires
custom CUDA kernels that process all tokens in a single kernel call.

## References

- vLLM Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- vLLM GitHub: https://github.com/vllm-project/vllm
- HuggingFace Qwen2: https://huggingface.co/docs/transformers/model_doc/qwen2
