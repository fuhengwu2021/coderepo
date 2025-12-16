# Pipeline Parallelism Demo

This directory contains a demonstration of **Pipeline Parallelism (PP)** for distributed inference.

## Overview

Pipeline Parallelism is a strategy where:
- Model layers are **split across multiple ranks** (stages)
- Each rank holds a **subset of layers**
- Activations flow **sequentially** through stages
- Enables running models that don't fit on a single GPU

## Key Concepts

1. **Stage Splitting**: Model layers are divided into stages, one per rank
2. **Sequential Processing**: Data flows through stages in a pipeline fashion
3. **Activation Passing**: Each stage receives activations from previous stage and sends to next
4. **Pipeline Bubbles**: Idle time during pipeline fill and drain phases
5. **Microbatches**: Splitting batches into smaller chunks to improve pipeline efficiency

## Files

- `parallel_state.py`: Manages pipeline parallel group initialization and communication
- `pipeline_model.py`: Pipeline stage and model splitting implementations
- `demo.py`: Main demo script showing various PP scenarios
- `__init__.py`: Package exports

## Usage

### Basic Usage

```bash
# Run with 2 processes (2 pipeline stages)
torchrun --nproc_per_node=2 demo.py

# Run with 4 processes (4 pipeline stages)
torchrun --nproc_per_node=4 demo.py

# Force CPU mode
torchrun --nproc_per_node=2 demo.py --force-cpu
```

## Demos

The demo script includes:

1. **Pipeline Parallel Forward Pass**: Shows how activations flow through stages
2. **Pipeline with Microbatches**: Demonstrates processing multiple microbatches
3. **Memory Benefits**: Shows how PP reduces memory per GPU
4. **Pipeline Efficiency**: Explains pipeline bubbles and efficiency considerations

## Pipeline Flow

```
Stage 0 (First)    Stage 1          Stage 2 (Last)
    |                |                  |
Input → [Layers] → Activations → [Layers] → Activations → [Layers] → Output
    |                |                  |
  Rank 0          Rank 1             Rank 2
```

## Comparison with Other Parallelism Strategies

- **vs Tensor Parallelism (TP)**: 
  - TP shards parameters within layers (fine-grained)
  - PP shards layers across ranks (coarse-grained)
  - TP requires communication during forward pass
  - PP requires communication between stages

- **vs Data Parallelism (DP)**:
  - DP replicates entire model on each rank
  - PP splits model across ranks
  - DP processes independent batches in parallel
  - PP processes batches sequentially through stages

## When to Use Pipeline Parallelism

- When model is too large to fit on a single GPU
- When you've maxed out efficient tensor parallelism
- For very deep models where layer distribution is efficient
- When you need to distribute across multiple nodes

## Limitations

- **Pipeline Bubbles**: Idle time during pipeline fill/drain reduces efficiency
- **Communication Overhead**: Activations must be transferred between stages
- **Sequential Processing**: Cannot fully parallelize like data parallelism
- **Complexity**: More complex to implement and debug than data parallelism

## Improving Pipeline Efficiency

1. **Microbatches**: Split batches into smaller chunks to fill pipeline
2. **1F1B (1 Forward 1 Backward)**: Overlap forward and backward passes
3. **Gradient Accumulation**: Accumulate gradients across microbatches
4. **Pipeline Parallel + Tensor Parallel**: Combine PP with TP for very large models

## Example: Combining PP with TP

For very large models, you can combine pipeline and tensor parallelism:

```python
# 8 GPUs total: 2 pipeline stages, 4 tensor parallel per stage
pipeline_parallel_size = 2
tensor_parallel_size = 4
```

This allows scaling to models that require many GPUs.

