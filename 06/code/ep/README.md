# Expert Parallelism (EP) Demo

This directory contains a simplified implementation of Expert Parallelism inspired by vLLM's design. It demonstrates how MoE (Mixture-of-Experts) models can distribute experts across multiple GPUs.

## Overview

Expert Parallelism (EP) is a technique that distributes different experts in MoE models across separate GPUs. Unlike Tensor Parallelism which shards weights, EP assigns entire experts to different GPUs, requiring all-to-all communication to route tokens to the correct expert GPUs.

## Key Concepts

### Expert Parallelism Flow

1. **Router**: All GPUs compute router logits to determine which experts each token should use
2. **Dispatch**: Tokens are sent to the correct expert GPUs using all-to-all communication
3. **Expert Computation**: Each GPU computes on tokens assigned to its experts
4. **Combine**: Results are gathered back using all-to-all communication

### Communication Pattern

```
Token 0 → Expert 0 (GPU 0) → Result 0
Token 1 → Expert 1 (GPU 1) → Result 1
Token 2 → Expert 0 (GPU 0) → Result 2
Token 3 → Expert 1 (GPU 1) → Result 3
```

Each GPU:
- Holds different experts
- Receives tokens that need its experts
- Computes expert outputs
- Sends results back

## Files

- `parallel_state.py`: Manages expert parallel group state and communication operations
- `moe.py`: Implements `ExpertParallelMoE` with expert parallelism
- `demo.py`: Demonstration script showing EP concepts

## Usage

### Running the Demo

```bash
# Run with 2 processes (uses GPU if 2+ GPUs available, otherwise CPU)
torchrun --nproc_per_node=2 demo.py

# Force CPU usage even if GPUs are available
torchrun --nproc_per_node=2 demo.py --force-cpu
```

## Benefits of Expert Parallelism

### 1. Memory Reduction

Each GPU stores only a subset of experts.

**Example**: 
- 8-expert MoE model
- With EP=2: Each GPU stores 4 experts
- **Result**: 2x memory reduction per GPU

### 2. Better Locality

Each expert is fully contained on one GPU, avoiding weight sharding overhead.

### 3. Scalability

Can scale to many GPUs by distributing more experts.

## Comparison with Tensor Parallelism

| Aspect | Tensor Parallelism | Expert Parallelism |
|--------|-------------------|-------------------|
| **Weight Sharding** | Shards each weight matrix | Assigns entire experts |
| **Communication** | All-reduce (every layer) | All-to-all (MoE layers only) |
| **Best For** | Dense models | MoE models |
| **Memory Pattern** | Sharded weights | Replicated attention, sharded experts |

## Real vLLM Implementation

In vLLM, EP is implemented and can be enabled with:

```bash
vllm serve <model> \
    --enable-expert-parallel \
    --data-parallel-size 8 \
    --all2all-backend pplx
```

**Note**: EP requires additional dependencies (DeepEP, pplx-kernels, DeepGEMM) and may not be fully stable for all model/quantization/hardware combinations.

## References

- vLLM Expert Parallel Deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- Chapter 6: Distributed Inference Fundamentals and vLLM

