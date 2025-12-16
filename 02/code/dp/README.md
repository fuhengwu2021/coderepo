# Data Parallelism Demo

This directory contains a demonstration of **Data Parallelism (DP)** for distributed inference.

## Overview

Data Parallelism is a strategy where:
- Each rank maintains a **full copy** of the model
- Different ranks process **different batches** of data
- No communication needed during forward pass (unlike Tensor Parallelism)
- Gradients are synchronized during training (averaged across ranks)

## Key Concepts

1. **Model Replication**: Each GPU/rank has a complete copy of the model
2. **Data Sharding**: Different ranks process different subsets of the dataset
3. **Gradient Synchronization**: During training, gradients are averaged across all ranks
4. **Independent Inference**: During inference, each rank can process requests independently

## Files

- `parallel_state.py`: Manages data parallel group initialization and communication
- `model.py`: Simple model implementations for demonstration
- `demo.py`: Main demo script showing various DP scenarios
- `__init__.py`: Package exports

## Usage

### Basic Usage

```bash
# Run with 2 processes
torchrun --nproc_per_node=2 demo.py

# Run with 4 processes
torchrun --nproc_per_node=4 demo.py

# Force CPU mode
torchrun --nproc_per_node=2 demo.py --force-cpu
```

## Demos

The demo script includes:

1. **Data Parallel Inference**: Shows how different ranks process different data
2. **Data Parallel Training**: Demonstrates gradient synchronization
3. **Transformer Data Parallel**: Shows DP with transformer models
4. **Throughput Benefits**: Demonstrates how DP increases total throughput

## Comparison with Other Parallelism Strategies

- **vs Tensor Parallelism (TP)**: 
  - TP shards model parameters across ranks (memory efficient)
  - DP replicates model (memory intensive but simpler)
  - TP requires communication during forward pass
  - DP requires no communication during inference

- **vs Pipeline Parallelism (PP)**:
  - PP splits model layers across ranks (stages)
  - DP replicates entire model
  - PP requires sequential processing (pipeline bubbles)
  - DP allows parallel processing of independent batches

## When to Use Data Parallelism

- When model fits on a single GPU
- When you want to increase throughput by processing more requests in parallel
- When you have multiple independent batches to process
- When simplicity is preferred over memory efficiency

## Limitations

- Requires enough memory to store full model on each rank
- Not suitable for very large models that don't fit on a single GPU
- For training, requires gradient synchronization overhead

