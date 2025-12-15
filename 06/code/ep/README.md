# Expert Parallelism (EP) Demo

This directory contains a simplified implementation of Expert Parallelism inspired by vLLM's design. It demonstrates how MoE (Mixture-of-Experts) models work on a single GPU and how they can be distributed across multiple GPUs using Expert Parallelism.

## Overview

**Mixture-of-Experts (MoE)** is an architecture where multiple specialized "expert" networks replace the standard feed-forward layer. During inference, only a subset of experts are activated per token, enabling large models with efficient computation.

**Expert Parallelism (EP)** is a technique that distributes different experts in MoE models across separate GPUs. Unlike Tensor Parallelism which shards weights, EP assigns entire experts to different GPUs, requiring all-to-all communication to route tokens to the correct expert GPUs.

---

## Part 1: MoE Architecture - Single GPU Perspective

This section describes how MoE works from a **single GPU perspective** using `microsoft/Phi-tiny-MoE-instruct` as a concrete example.

### Model Architecture Overview

**Phi-tiny-MoE-instruct** is a 3.76B parameter MoE model with:
- **32 decoder layers** (each layer contains self-attention + MoE feed-forward)
- **16 experts** per MoE layer
- **Top-2 routing** (each token uses 2 experts)
- **Hidden dimension**: 4096
- **Expert intermediate size**: 448

### MoE Layer Structure (Single GPU)

From a single GPU's point of view, here's what happens in each MoE layer:

```
Input Token (hidden_size=4096)
    ↓
┌─────────────────────────────────────┐
│  Router (Gate Network)              │
│  Linear(4096 → 16)                  │
│  - Computes router logits for      │
│    all 16 experts                   │
│  - Selects top-2 experts per token │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Expert Selection (Top-2)          │
│  - Token 0 → Experts [3, 7]       │
│  - Token 1 → Experts [1, 12]      │
│  - Token 2 → Experts [3, 9]       │
│  ...                                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Expert Processing                  │
│  Each expert is a 3-layer MLP:     │
│  - w1: Linear(4096 → 448)          │
│  - w2: Linear(448 → 4096)           │
│  - w3: Linear(4096 → 448)           │
│  - Activation: SiLU                 │
│                                     │
│  Formula:                           │
│  output = w2(SiLU(w1(x)) * SiLU(w3(x)))│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Weighted Combination               │
│  output = Σ(weight_i * expert_i(x)) │
│  where i ∈ top-2 selected experts   │
└─────────────────────────────────────┘
    ↓
Output Token (hidden_size=4096)
```

### Key Components

#### 1. Router (Gate Network)
- **Structure**: Single linear layer `Linear(4096 → 16)`
- **Function**: Computes routing scores for all 16 experts
- **Output**: Router logits `[num_tokens, 16]`
- **Selection**: Top-2 experts per token based on logits

#### 2. Expert Architecture
Each of the 16 experts is a **SwiGLU MLP**:
```
Expert Structure:
  Input (4096) 
    → w1 (4096 → 448) → SiLU
    → w3 (4096 → 448) → SiLU
    → Element-wise multiply
    → w2 (448 → 4096)
  Output (4096)
```

#### 3. Token-to-Expert Routing
- Each token gets routed to **2 experts** (top-2)
- Router weights are softmax-normalized
- Final output = weighted sum of 2 expert outputs

### Forward Pass Flow (Single GPU)

1. **Router computation**:
   ```python
   router_logits = gate(input)  # [num_tokens, 16]
   topk_weights, topk_ids = topk(router_logits, k=2)
   ```

2. **Expert computation**:
   ```python
   for token in tokens:
       output = 0
       for expert_id, weight in zip(selected_experts, weights):
           output += weight * experts[expert_id](token)
   ```

3. **Result**: Weighted sum of expert outputs

### Example: Processing 4 Tokens (Single GPU)

```
Token 0: Router selects Experts [3, 7]  → weights [0.6, 0.4]
Token 1: Router selects Experts [1, 12] → weights [0.7, 0.3]
Token 2: Router selects Experts [3, 9]  → weights [0.5, 0.5]
Token 3: Router selects Experts [7, 15] → weights [0.8, 0.2]

Output 0 = 0.6 * Expert3(Token0) + 0.4 * Expert7(Token0)
Output 1 = 0.7 * Expert1(Token1) + 0.3 * Expert12(Token1)
Output 2 = 0.5 * Expert3(Token2) + 0.5 * Expert9(Token2)
Output 3 = 0.8 * Expert7(Token3) + 0.2 * Expert15(Token3)
```

All experts are stored and computed on the same GPU.

### Memory and Computation on Single GPU

**Single GPU (all experts on one GPU)**:
- **16 experts** × **~55M params per expert** = **~880M params per MoE layer**
- **32 layers** × **~880M** = **~28B params** (just for MoE layers)
- **Total model**: ~3.76B params (includes attention, embeddings, etc.)
- **Active during inference**: Only 2 experts per token (~1.1B active params)

### Why Top-2 Routing?

- **Efficiency**: Only 2 experts activated per token (12.5% of total experts)
- **Active parameters**: ~1.1B params active during inference (vs 3.76B total)
- **Quality**: Top-2 provides good balance between specialization and coverage
- **Load balancing**: Better distribution of tokens across experts

---

## Part 2: Expert Parallelism - Scaling to Multiple GPUs

When a single GPU cannot hold all experts, **Expert Parallelism (EP)** distributes experts across multiple GPUs.

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

### Example: 4 GPUs with 16 Experts

**Expert Distribution**:
- **GPU 0**: Experts [0, 1, 2, 3]
- **GPU 1**: Experts [4, 5, 6, 7]
- **GPU 2**: Experts [8, 9, 10, 11]
- **GPU 3**: Experts [12, 13, 14, 15]

**Forward Pass with EP**:

1. **Router computation** (replicated on all GPUs):
   ```python
   router_logits = gate(input)  # [num_tokens, 16]
   topk_weights, topk_ids = topk(router_logits, k=2)
   ```

2. **Token dispatch** (all-to-all communication):
   - Token 0 needs Experts [3, 7] → sent to GPU 0 and GPU 1
   - Token 1 needs Experts [1, 12] → sent to GPU 0 and GPU 3
   - Token 2 needs Experts [3, 9] → sent to GPU 0 and GPU 2
   - etc.

3. **Expert computation** (on each GPU):
   ```python
   # On GPU 0 (has experts [0,1,2,3])
   for token in tokens_assigned_to_gpu0:
       for expert_id in token.selected_experts:
           if expert_id in [0,1,2,3]:  # Local experts
               output += weight * local_experts[expert_id](token)
   ```

4. **Result combination** (all-to-all communication):
   - Expert outputs are gathered back to original GPUs
   - Weighted sum produces final token representation

### Memory and Computation with EP

**With Expert Parallelism (EP=4, 4 experts per GPU)**:
- **4 experts** × **~55M params** = **~220M params per MoE layer per GPU**
- **4x memory reduction** per GPU for MoE layers
- Each GPU only stores and computes its assigned experts

**Comparison**:
| Configuration | Experts per GPU | Memory per GPU (MoE layers) |
|---------------|----------------|----------------------------|
| Single GPU    | 16             | ~880M params               |
| EP=2          | 8              | ~440M params               |
| EP=4          | 4              | ~220M params               |
| EP=8          | 2              | ~110M params               |

---

## Benefits of Expert Parallelism

### 1. Memory Reduction

Each GPU stores only a subset of experts.

**Example**: 
- 16-expert MoE model
- With EP=4: Each GPU stores 4 experts
- **Result**: 4x memory reduction per GPU

### 2. Better Locality

Each expert is fully contained on one GPU, avoiding weight sharding overhead.

### 3. Scalability

Can scale to many GPUs by distributing more experts. For example:
- 16 experts → 4 GPUs (4 experts each)
- 64 experts → 16 GPUs (4 experts each)
- 128 experts → 32 GPUs (4 experts each)

### 4. Efficient Communication

Communication only happens at MoE layers (not every layer like Tensor Parallelism), and uses all-to-all which can be optimized.

---

## Comparison with Tensor Parallelism

| Aspect | Tensor Parallelism | Expert Parallelism |
|--------|-------------------|-------------------|
| **Weight Sharding** | Shards each weight matrix | Assigns entire experts |
| **Communication** | All-reduce (every layer) | All-to-all (MoE layers only) |
| **Best For** | Dense models | MoE models |
| **Memory Pattern** | Sharded weights | Replicated attention, sharded experts |
| **Communication Frequency** | Every layer | Only MoE layers |

---

## Files

- `parallel_state.py`: Manages expert parallel group state and communication operations
- `moe.py`: Implements `ExpertParallelMoE` with expert parallelism
- `demo.py`: Demonstration script showing EP concepts
- `moe_arch.py`: Example script loading Phi-tiny-MoE-instruct to inspect MoE architecture

## Usage

### Running the Demo

```bash
# Run with 2 processes (uses GPU if 2+ GPUs available, otherwise CPU)
torchrun --nproc_per_node=2 demo.py

# Force CPU usage even if GPUs are available
torchrun --nproc_per_node=2 demo.py --force-cpu
```

### Inspecting MoE Architecture

```bash
# Load and inspect Phi-tiny-MoE-instruct MoE structure
python moe_arch.py
```

---

## Real vLLM Implementation

In vLLM, EP is implemented and can be enabled with:

```bash
vllm serve <model> \
    --enable-expert-parallel \
    --data-parallel-size 8 \
    --all2all-backend pplx
```

**Note**: EP requires additional dependencies (DeepEP, pplx-kernels, DeepGEMM) and may not be fully stable for all model/quantization/hardware combinations.

---

## References

- vLLM Expert Parallel Deployment: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- Chapter 6: Distributed Inference Fundamentals and vLLM
