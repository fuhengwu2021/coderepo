# PagedAttention v4: True Ragged Batching Implementation

## Overview

PagedAttention v4 implements **ragged batching metadata structure** as described in a6.md. While we demonstrate the metadata organization (flattened tokens + metadata arrays), the actual processing still uses sequential calls due to HuggingFace limitations.

## What We Implemented

### 1. Ragged Batching Metadata

**Prefill Phase**:
- Flatten all prompt tokens: `T = sum(L_i)` instead of `B × max(L_i)`
- Build metadata arrays:
  - `token_ids_flat[T]`: All token IDs
  - `seq_id_flat[T]`: Which sequence each token belongs to
  - `position_flat[T]`: Position of each token in its sequence
  - `slot_mapping_flat[T]`: Physical slot for each token's KV

**Decode Phase**:
- Each sequence has 1 query token
- Flatten to `num_seqs × 1` (not `B × Lmax`)
- Use metadata to identify each token

### 2. Prefill Batching

```python
# Collect multiple requests
seq_ids, prompt_token_lists, positions_start, seq_lengths = scheduler.get_prefill_batch()

# Build ragged metadata
token_ids_flat, seq_id_flat, position_flat, slot_mapping_flat = \
    build_ragged_metadata(seq_ids, prompt_token_lists, positions_start)

# T = total tokens (not B × max_len)
T = len(token_ids_flat)  # e.g., 109 tokens for 3 sequences
# Would be 111 with padding (3 × 37), saving 2 tokens
```

### 3. Metadata Structure

Example with 3 sequences, prompt lengths [36, 37, 36]:

```
token_ids_flat:    [tok0, tok1, ..., tok35, tok0, tok1, ..., tok36, tok0, ..., tok35]
seq_id_flat:       [0, 0, ..., 0,           1, 1, ..., 1,           2, 2, ..., 2]
position_flat:     [0, 1, ..., 35,          0, 1, ..., 36,          0, 1, ..., 35]
slot_mapping_flat: [slot0, slot1, ..., slot35, slot4, slot5, ..., slot40, ...]
```

## Limitations

### What We Actually Do

1. ✅ **Metadata Structure**: Build flattened tokens + metadata arrays
2. ✅ **Prefill Batching**: Collect multiple requests, process together
3. ✅ **No Padding**: Each sequence uses only needed blocks
4. ❌ **True Ragged Processing**: Still process sequences separately (HuggingFace limitation)

### Why Not True Ragged Batching?

HuggingFace's `model.forward()` doesn't support true ragged batching:
- It expects standard batch format `[B, L]` or `[B, L, D]`
- Cannot handle flattened `[T]` with metadata
- Would treat flattened tokens as one long sequence

### What True Ragged Batching Would Require

1. **Custom Attention Kernels**: Process flattened tokens `[T, D]` with metadata
2. **Attention Mask from Metadata**: Use `seq_id_flat` to build causal masks per sequence
3. **Single Forward Pass**: All tokens processed in one kernel call
4. **No Python Loops**: Everything in CUDA kernels

## Code Structure

### Key Methods

**`build_ragged_metadata()`**: 
- Flattens tokens from multiple sequences
- Builds metadata arrays (seq_id, position, slot_mapping)

**`prefill_batch()`**:
- Collects sequences ready for prefill
- Builds metadata
- Processes sequences (currently separate, but demonstrates metadata)

**`decode_batch()`**:
- Processes multiple sequences in decode phase
- Uses metadata to identify tokens

## Example Output

```
[Prefill Batch] Processing 3 sequences, 109 total tokens 
(would be 111 with padding, saving 2 tokens)
[Prefill Batch] Metadata: seq_id_flat=3 sequences, 
position_flat=109 tokens, slot_mapping_flat=109 slots
[Prefill Batch] Note: Using metadata structure, but processing sequences separately
```

## Comparison with vLLM

| Feature | v4 (Our Implementation) | vLLM |
|---------|-------------------------|------|
| Metadata Structure | ✅ | ✅ |
| Flattened Tokens | ✅ | ✅ |
| Prefill Batching | ✅ (with metadata) | ✅ |
| Single Kernel Call | ❌ (HuggingFace limitation) | ✅ |
| True Ragged Processing | ❌ | ✅ |
| Custom CUDA Kernels | ❌ | ✅ |

## What This Demonstrates

1. **Metadata Organization**: How to structure flattened tokens + metadata
2. **Prefill Batching**: How to batch multiple prefill requests
3. **No Padding**: How ragged batching eliminates padding tokens
4. **Concept**: The structure needed for true ragged batching

## Next Steps for True Implementation

To achieve true ragged batching, you would need:

1. Custom attention kernels that accept:
   - `hidden_states[T, D]`: Flattened hidden states
   - `seq_id_flat[T]`: Sequence ID for each token
   - `position_flat[T]`: Position in sequence
   - `block_tables`: Per-sequence block mappings

2. Attention mask construction from metadata:
   - Build causal masks per sequence using `seq_id_flat`
   - Ensure tokens only attend to tokens in same sequence

3. Single forward pass:
   - Process all `T` tokens in one kernel call
   - Use metadata to route results to correct sequences

This is what vLLM does with its custom CUDA kernels.
