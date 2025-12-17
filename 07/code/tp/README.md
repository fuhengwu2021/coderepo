# Tensor Parallelism (TP) with RadixAttention

This directory contains a simplified implementation of Tensor Parallelism that integrates with RadixAttention, following SGLang's approach.

## Overview

This implementation demonstrates how TP works with RadixAttention:

1. **TP is applied to attention layers**:
   - QKV projection uses **column parallelism** (splits along output dimension)
   - Output projection uses **row parallelism** (splits along input dimension)

2. **RadixAttention works with TP-sharded tensors**:
   - Q, K, V are already sharded per TP rank
   - RadixAttention operates on sharded tensors
   - All-reduce happens in output projection

3. **RadixCache coordination**:
   - Main rank (rank 0) manages RadixCache
   - All ranks participate in computation
   - Cache is shared conceptually (simplified implementation)

## Key Differences from Chapter 6 TP

Chapter 6's TP is a **generic TP implementation** inspired by vLLM:
- Standalone TP for educational purposes
- Works with any model architecture
- No integration with RadixAttention or scheduler

This implementation is **SGLang-specific**:
- Integrates with RadixAttention from `../radix/`
- Works with scheduler for request coordination
- Follows SGLang's TP patterns (column/row parallelism for attention)
- RadixCache-aware (prefix sharing works with TP)

## Files

- `parallel_state.py`: TP group management and communication operations
- `linear.py`: ColumnParallelLinear and RowParallelLinear layers
- `tp_model_wrapper.py`: TP-aware RadixAttention model wrapper
- `scheduler.py`: Simple scheduler for coordinating TP workers
- `demo_tp_radix.py`: Demo script showing TP + RadixAttention

## Usage

### Running the Demo

```bash
# Run with 2 TP ranks (requires 2+ GPUs or CPU)
torchrun --nproc_per_node=2 demo_tp_radix.py

# Run with 4 TP ranks
torchrun --nproc_per_node=4 demo_tp_radix.py

# Force CPU usage (if no GPUs available)
torchrun --nproc_per_node=2 demo_tp_radix.py --force-cpu

# Specify model
torchrun --nproc_per_node=2 demo_tp_radix.py --model-name Qwen/Qwen2.5-0.5B-Instruct
```

### Using in Your Code

```python
import torch
import torch.distributed as dist
from tp.parallel_state import initialize_tensor_parallel
from tp.tp_model_wrapper import TPRadixAttentionModelWrapper
from tp.scheduler import TPScheduler

# Initialize distributed
dist.init_process_group(backend="nccl")
initialize_tensor_parallel(tensor_parallel_size=2)

# Create TP RadixAttention model
model_wrapper = TPRadixAttentionModelWrapper(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda",
    page_size=16,
)

# Create scheduler
scheduler = TPScheduler(model_wrapper)

# Add requests
request_id = scheduler.add_request("What is AI?", max_new_tokens=20)

# Process requests
results = scheduler.process_requests()
```

## How It Works

### 1. TP Initialization

```python
initialize_tensor_parallel(tensor_parallel_size=2)
```

This creates TP groups and sets up communication:
- Each TP rank gets a subset of attention heads
- Communication groups for all-reduce and all-gather

### 2. Model Wrapper

`TPRadixAttentionModelWrapper`:
- Loads the model on each TP rank
- Applies TP to attention layers (replaces QKV and output projections)
- Integrates with RadixCache (managed on main rank)
- All ranks participate in forward pass

### 3. Attention with TP

For each attention layer:
- **QKV projection**: Column parallel → each rank gets `num_heads / tp_size` heads
- **Attention computation**: Works with sharded Q/K/V
- **Output projection**: Row parallel → all-reduce to combine results

### 4. RadixCache Integration

- Main rank (rank 0) manages RadixCache
- Prefix matching happens on main rank
- All ranks compute, but cache coordination is simplified
- In full SGLang, cache is coordinated across ranks

## Limitations

This is a **simplified educational implementation**:

1. **Cache coordination**: RadixCache is only managed on main rank. In full SGLang, cache is coordinated across TP ranks.

2. **Batching**: Scheduler processes one request at a time. Full SGLang has sophisticated batching.

3. **Communication**: Uses basic PyTorch distributed ops. SGLang has optimized communication.

4. **Model loading**: Each rank loads full model then shards. Full SGLang loads sharded weights directly.

5. **RadixCache sharing**: Simplified - main rank manages cache. Full implementation would coordinate across ranks.

## Comparison with SGLang

| Feature | This Implementation | SGLang |
|---------|-------------------|--------|
| TP for attention | ✅ Column/Row parallel | ✅ Same |
| RadixAttention | ✅ Integrated | ✅ Same |
| Cache coordination | ⚠️ Main rank only | ✅ Coordinated |
| Batching | ⚠️ Single request | ✅ Continuous batching |
| Communication | ⚠️ Basic PyTorch | ✅ Optimized kernels |
| Weight loading | ⚠️ Load then shard | ✅ Sharded loading |

## References

- SGLang source code: `resources/sglang2025/`
- Chapter 6 TP: `../chapter6-distributed-inference-fundamentals-and-vllm/code/tp/`
- RadixAttention: `../radix/`
- Chapter 7: Distributed Inference with SGLang
