# Tensor Parallelism (TP) Demo

This directory contains a simplified implementation of Tensor Parallelism inspired by vLLM's design. It demonstrates the core concepts of column parallelism, row parallelism, and how they work together in an MLP layer.

## Overview

Tensor Parallelism (TP) is a technique that shards model weights horizontally across multiple GPUs within a single node, allowing all GPUs to compute concurrently. This follows an **SPMD (Single Program, Multiple Data)** paradigm.

## Key Concepts

### Column Parallelism

Splits the weight matrix along its **columns**:

```
Y = X × A

Split A into [A₁ | A₂] (column-wise)
Then: Y = [X × A₁ | X × A₂]
```

- Each GPU computes one piece of the result
- Use **all-gather** operation to concatenate the pieces (optional)
- Result: Sharded output vector on each GPU (or full output if gathered)

### Row Parallelism

Splits the weight matrix along its **rows**:

```
Y = X × A

Split X into [X₁; X₂] and A into [A₁; A₂] (row-wise)
Then: Y = X₁ × A₁ + X₂ × A₂
```

- Each GPU computes a partial result
- Use **all-reduce** operation to sum the partial results
- Result: Final output vector on all GPUs

### MLP with Tensor Parallelism

An MLP layer consists of:

```
Input → Up Projection (Column Parallel) → Activation → Down Projection (Row Parallel) → Output
```

**Key Insight**: Within an MLP, we can avoid the all-gather after column parallel because the next operation (row parallel) needs sharded data anyway. We only need all-reduce at the end.

## Files

- `parallel_state.py`: Manages tensor parallel group state and communication operations
- `linear.py`: Implements `ColumnParallelLinear` and `RowParallelLinear` layers
- `mlp.py`: Implements `TensorParallelMLP` that combines column and row parallelism
- `demo.py`: Demonstration script showing all concepts

## Dependencies

This demo uses the shared `mdaisy` utility package for distributed initialization:
- Located at: `resources/coderepo/shared/mdaisy`
- Install with: `pip install -e resources/coderepo/shared`
- Or ensure the path is available (the demo will try to add it automatically)

## Usage

### Running the Demo

```bash
# Run with 2 processes (uses GPU if 2+ GPUs available, otherwise CPU)
torchrun --nproc_per_node=2 demo.py

# Run with 4 processes
torchrun --nproc_per_node=4 demo.py

# Force CPU usage even if GPUs are available
torchrun --nproc_per_node=2 demo.py --force-cpu
```

**Note**: The demo automatically uses CPU when:
- No GPUs are available, OR
- Less than 2 GPUs are available (needed for tensor parallelism)

You can also force CPU usage with the `--force-cpu` flag.

**Quick CPU Test**:
```bash
# Test with CPU (requires 2 processes)
./test_cpu.sh
# or
torchrun --nproc_per_node=2 demo.py --force-cpu
```

### Using in Your Code

```python
import torch
import torch.distributed as dist
from parallel_state import initialize_tensor_parallel
from mlp import TensorParallelMLP

# Determine device and backend
use_cpu = not torch.cuda.is_available() or torch.cuda.device_count() < 2
backend = "gloo" if use_cpu else "nccl"
device = torch.device("cpu" if use_cpu else f"cuda:{dist.get_rank()}")

# Initialize distributed environment
dist.init_process_group(backend=backend)
initialize_tensor_parallel(tensor_parallel_size=2, backend=backend)

# Create tensor parallel MLP
mlp = TensorParallelMLP(hidden_size=128, intermediate_size=512).to(device)

# Forward pass
x = torch.randn(4, 10, 128, device=device)
output = mlp(x)  # Output is the same on all processes after all-reduce
```

## Benefits of Tensor Parallelism

### 1. Weight Space Reduction

Each GPU stores only a fraction of the weights.

**Example**: 
- 140B parameter model
- With TP=2: Each GPU stores ~70B parameters
- **Result**: Model can now fit on GPUs that couldn't hold the full model

### 2. KV Cache Space Increase

More space available for KV cache per GPU.

**Example**:
- Single GPU: 160GB total, 140GB for weights → 20GB for KV cache
- With TP=2: Each GPU has 160GB, 70GB for weights → 90GB for KV cache
- **Result**: Super-linear increase in KV cache capacity

### 3. Latency Reduction

Faster computation and memory bandwidth utilization.

**Mechanism**:
- Each GPU loads fewer weights from HBM to compute
- Effectively multiplies memory bandwidth
- Prefill operations (often memory-bound) benefit significantly

**Trade-off**: Communication overhead between GPUs

### 4. Communication Cost

**Data transferred per layer**:
- Size: `batch_size × sequence_length × hidden_size`
- Occurs for both MLP and attention layers
- Repeated for every layer in the model

**Mitigation**: Good communication hardware (e.g., NVLink within a node) reduces this overhead.

## Implementation Details

### ColumnParallelLinear

- Weight matrix `A` is sharded along columns: `[A_1, A_2, ..., A_p]`
- Each GPU stores `[input_size, output_size / tp_size]`
- Forward: `output = input @ weight_shard` (sharded output)
- Optional all-gather to get full output on all GPUs

### RowParallelLinear

- Weight matrix `A` is sharded along rows: `[A_1; A_2; ...; A_p]`
- Each GPU stores `[input_size / tp_size, output_size]`
- Forward: `output = input_shard @ weight_shard` (partial result)
- All-reduce to sum partial results and get final output

### Communication Operations

- **All-gather**: Concatenates results from multiple GPUs
- **All-reduce**: Sums partial results from multiple GPUs

## Key Features (vLLM-style)

This implementation follows vLLM's tensor parallelism approach:

- **Real Weight Loading**:
  - Loads weights from HuggingFace checkpoints
  - Shards weights **during loading** (not after)
  - Each TP rank only stores its shard in memory
  - Matches vLLM's production behavior

- **Proper Weight Loaders**:
  - `weight_loader()` methods for ColumnParallel and RowParallel layers
  - `weight_loader_qkv()` for fused QKV projections
  - Handles both fused and separate Q/K/V formats
  - Supports GQA/MQA (different KV heads)

- **QKVParallelLinear**:
  - Handles fused QKV projections
  - Proper head sharding for GQA
  - Matches vLLM's QKVParallelLinear API

- **Inference Optimizations**:
  - Uses `torch.inference_mode()` for better performance
  - No autograd overhead
  - Proper dtype handling

## Comparison with Generic TP

**Generic TP (tp/ directory)**:
- Random weight initialization
- Weights sharded after creation
- Educational/demonstration purposes

**vLLM-style TP (tp-vllm/ directory)**:
- Real weight loading from checkpoints
- Weights sharded during loading
- Production-ready approach
- Matches vLLM's actual implementation

## Files

- `parallel_state.py`: TP group management (same as generic TP)
- `linear.py`: ColumnParallelLinear, RowParallelLinear, **QKVParallelLinear** (vLLM-style)
- `mlp.py`: TensorParallelMLP with proper weight loading
- `attention.py`: TensorParallelAttention layer
- `weight_loader.py`: Utilities for loading and sharding weights
- `model_wrapper.py`: Wrapper that loads real model weights
- `demo.py`: Basic TP demo (same as generic TP)
- `demo_vllm.py`: **vLLM-style demo with real weight loading**

## Usage

### Basic Demo (Generic TP)

```bash
# Run basic TP demo (random initialization)
torchrun --nproc_per_node=2 demo.py --force-cpu
```

### vLLM-style Demo (Real Weight Loading)

```bash
# Run vLLM-style demo with real weight loading
torchrun --nproc_per_node=2 demo_vllm.py --force-cpu
```

This demonstrates:
- Loading weights from HuggingFace checkpoint
- Sharding weights during loading (not after)
- Proper weight_loader methods
- QKVParallelLinear for attention
- Memory savings per TP rank

## Implementation Details

### Weight Loading (vLLM-style)

The key difference from generic TP is **when** weights are sharded:

**Generic TP:**
1. Initialize random weights
2. Shard weights after creation
3. Each rank stores shard

**vLLM-style TP:**
1. Load full weights from checkpoint
2. **Shard during loading** (using weight_loader)
3. Each rank only stores its shard in memory

This matches vLLM's production approach where:
- Main rank (or all ranks) loads checkpoint
- Weights are sharded immediately using `weight_loader()` methods
- Memory footprint is reduced from the start

### QKVParallelLinear

Handles fused QKV projections with proper head sharding:

- **Fused QKV**: Single weight matrix `[hidden_size, (Q+K+V)*head_size]`
- **Separate Q/K/V**: Three separate weight matrices
- **GQA Support**: Handles different number of KV heads vs Q heads
- **Proper Sharding**: Q heads partitioned, KV heads partitioned or replicated

### Weight Loaders

Each TP layer has a `weight_loader()` method:

- `ColumnParallelLinear.weight_loader()`: Shards along output dimension
- `RowParallelLinear.weight_loader()`: Shards along input dimension  
- `QKVParallelLinear.weight_loader_qkv()`: Handles Q/K/V sharding

## Source Code Location

Key files matching vLLM's structure:

- **ColumnParallelLinear**: `linear.py` - Matches `vllm/model_executor/layers/linear.py`
- **RowParallelLinear**: `linear.py` - Matches vLLM's RowParallelLinear
- **QKVParallelLinear**: `linear.py` - Matches vLLM's QKVParallelLinear
- **Weight Loading**: `weight_loader.py` - Simplified version of vLLM's weight loading
- **Attention**: `attention.py` - TP-aware attention layer

## References

- vLLM source code: `resources/vllm/vllm/model_executor/layers/linear.py`
- vLLM weight loading: `resources/vllm/vllm/model_executor/parameter.py`
- Chapter 6: Distributed Inference Fundamentals and vLLM

