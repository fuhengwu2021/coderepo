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

## Comparison with vLLM

This implementation is a simplified version of vLLM's tensor parallelism:

- **Similarities**:
  - Same column/row parallelism concepts
  - Same communication patterns (all-gather, all-reduce)
  - Same MLP structure (column → activation → row)

- **Simplifications**:
  - No quantization support
  - No custom op registration
  - Simpler weight loading
  - Basic initialization (no advanced optimizations)

## References

- vLLM source code: `resources/vllm/vllm/model_executor/layers/linear.py`
- Chapter 6: Distributed Inference Fundamentals and vLLM

