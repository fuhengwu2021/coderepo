# PagedAttention v3: Continuous Batching Implementation

## Overview

PagedAttention v3 extends v2 with **continuous batching** support, allowing multiple sequences to be processed concurrently without padding.

## Key Features

### 1. Continuous Batching
- **Multiple sequences processed concurrently**: Different sequences can be in prefill or decode state simultaneously
- **Dynamic scheduling**: Scheduler manages which sequences to process in each step
- **No padding**: Each sequence processes only its actual tokens (ragged batching)

### 2. Scheduler (`scheduler.py`)
- **Sequence state management**: Tracks PREFILL, DECODE, FINISHED states
- **Batch construction**: Selects sequences ready for processing
- **Dynamic updates**: Updates sequences after each step

### 3. Batch Attention Computation
- **`compute_attention_batch`**: Processes multiple sequences in one call
- **Ragged batching**: Each sequence has its own block_table and position
- **Efficient**: Reuses v2's online/safe softmax algorithms

## Architecture

```
ContinuousBatchScheduler
│
├── SequenceInfo (per sequence)
│   ├── state: PREFILL | DECODE | FINISHED
│   ├── prompt_tokens: List[int]
│   ├── generated_tokens: List[int]
│   └── position: int
│
└── get_batch() → (seq_ids, positions, token_ids)
    └── Returns sequences ready for current step

PagedAttentionV3 (per layer)
│
├── compute_attention_batch(seq_ids, q_batch, positions)
│   └── Processes multiple sequences concurrently
│
└── append_kv_batch(seq_ids, k_batch, v_batch, positions)
    └── Caches KV for multiple sequences
```

## Usage Example

```python
from pa import PagedAttentionV3
from pa.scheduler import ContinuousBatchScheduler

# Initialize model wrapper
model_wrapper = PagedAttentionModelWrapperV3(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    max_batch_size=32
)

# Add multiple requests
seq_id1 = model_wrapper.add_request("What is AI?", max_new_tokens=50)
seq_id2 = model_wrapper.add_request("Explain ML.", max_new_tokens=50)

# Process continuous batch
while model_wrapper.scheduler.get_active_count() > 0:
    seq_ids, next_tokens = model_wrapper.step()
    # Sequences are processed concurrently
    
# Get results
text1 = model_wrapper.get_sequence_text(seq_id1)
text2 = model_wrapper.get_sequence_text(seq_id2)
```

## How It Works

### What We Actually Implemented

**Important**: This is **NOT true ragged batching**. We implement:

1. **Sequential Pseudo-Batching**: 
   - Scheduler selects multiple sequences ready for processing
   - But we process them sequentially using Python loops
   - Each sequence calls `compute_attention` independently

2. **Shared Block Pool**: 
   - Multiple sequences share the same block pool
   - No padding needed (each sequence uses only needed blocks)
   - Blocks are allocated/deallocated per sequence

3. **Concurrent Scheduling**:
   - Scheduler manages which sequences are in prefill/decode state
   - Multiple sequences can be "in batch" but processed sequentially

### What True Ragged Batching Would Be

True ragged batching (like vLLM) would:
1. **Flatten tokens**: All sequences' tokens → `T×D` (T = total tokens)
2. **Metadata arrays**: `seq_id[T]`, `position[T]`, `slot_mapping[T]`
3. **Single kernel call**: Process all tokens at once
4. **No Python loops**: Everything in CUDA kernels

### Step-by-Step Flow (Current Implementation)

1. **Add Requests**: Each request is added to scheduler and prefill is done immediately
2. **Pseudo-Batching Loop**:
   - Scheduler selects sequences ready for decode
   - **Sequentially** process each sequence (Python loop)
   - Updates scheduler with generated tokens
   - Cleans up finished sequences
3. **No Padding**: Each sequence uses only the blocks it needs (this part is correct)

### Example: 3 Sequences Concurrently

```
Step 1-10: All 3 sequences in decode (batch_size=3)
Step 11: Sequence 0 finishes (batch_size=2)
Step 12-20: Sequences 1, 2 in decode (batch_size=2)
Step 21: Sequence 1 finishes (batch_size=1)
Step 22-30: Sequence 2 in decode (batch_size=1)
Step 31: Sequence 2 finishes (batch_size=0, done)
```

## Differences from v2

| Feature | v2 | v3 |
|---------|----|----|
| **Batching** | Single sequence | Multiple sequences concurrently |
| **Scheduling** | None | ContinuousBatchScheduler |
| **Prefill** | Per-sequence | Batch prefill (immediate) |
| **Decode** | Sequential | Concurrent decode steps |
| **Memory** | Per-sequence blocks | Shared block pool across sequences |

## Limitations

This is a **simplified educational implementation**. Key limitations:

1. **NOT true ragged batching**: Uses Python loops to process sequences sequentially
2. **NOT kernel-level batching**: No custom CUDA kernels for batch attention
3. **Sequential processing**: Even though scheduler selects multiple sequences, we process them one-by-one
4. **No prefill batching**: Prefill is done immediately per request (not batched)

Production systems like vLLM have:

1. **True ragged batching**: Flattened tokens `T×D` + metadata arrays
2. **Kernel-level batching**: Custom CUDA kernels process all tokens in one call
3. **Advanced scheduling**: Priority queues, latency budgets, etc.
4. **Prefill batching**: Multiple prefill sequences in same batch

## What We Actually Achieved

Despite not being true ragged batching, v3 provides:

1. ✅ **Concurrent sequence management**: Scheduler tracks multiple sequences
2. ✅ **Shared block pool**: Multiple sequences share blocks (no padding)
3. ✅ **Dynamic scheduling**: Sequences can be added/removed dynamically
4. ✅ **State management**: Proper prefill → decode → finished transitions
5. ❌ **True parallel processing**: Still sequential (Python loops)
6. ❌ **Ragged batching**: Not implemented (would need flattened tokens + metadata)

## Files

- **`scheduler.py`**: Continuous batching scheduler
- **`paged_attention_v3.py`**: PagedAttention v3 with batch support
- **`inference_v3.py`**: Example inference script with continuous batching

## Testing

```bash
python inference_v3.py
```

This will:
1. Add 3 requests to the batch
2. Process them concurrently
3. Show generation results for all sequences
4. Display scheduler and block statistics
