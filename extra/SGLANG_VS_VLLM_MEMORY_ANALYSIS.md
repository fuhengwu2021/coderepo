# SGLang vs vLLM Memory Usage Root Cause Analysis

## Problem Statement

**SGLang cannot start with 10M context length** on 8x H200 GPUs, while **vLLM successfully supports 10M context** with the same hardware. Both use FP8 E4M3 KV cache and similar optimizations, yet SGLang fails with OOM during model loading.

## Root Cause: Memory Allocation Strategy Differences

### 1. **KV Cache Allocation Strategy**

#### vLLM: **Dynamic Paged Allocation**
- **Strategy**: KV cache is **paged/managed dynamically**, not pre-allocated in full
- **Allocation**: Allocates KV cache blocks **on-demand** as sequences grow
- **Memory Formula**: `gpu-memory-utilization` (default 0.9) controls the **total budget**, but KV cache is allocated incrementally
- **Key Point**: vLLM does **NOT** pre-allocate KV cache for the full `max-model-len` at startup
- **Evidence**: From `HYBRID_KV_CACHE_ANALYSIS.md`:
  > "vLLM 的 KV Cache 是分页管理的，不是一次性全量分配"
  > "KV cache 按可用显存预算预留/分页管理"
  > "不是按 `max_model_len` 把 2M tokens 一次性分配到每层"

#### SGLang: **Static Pre-allocation**
- **Strategy**: KV cache pool is **pre-allocated statically** during initialization
- **Allocation**: Allocates KV cache pool upfront based on `context-length` and `mem-fraction-static`
- **Memory Formula**: `mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity`
- **Key Point**: SGLang **pre-allocates** KV cache pool for the full `context-length` at startup
- **Evidence**: From `server_args.py:716`:
  > "The argument mem_fraction_static is defined as (model weights + KV cache pool) / GPU memory capacity"

### 2. **Memory Calculation During Model Loading**

#### vLLM Memory Allocation (10M context):
```
Total GPU Memory: 140 GB
gpu-memory-utilization: 0.95 (for 10M test)
Available Budget: 140 GB × 0.95 = 133 GB

During Model Loading:
1. Model Weights: ~4 GB (distributed across 8 GPUs)
2. KV Cache Pool: Allocated dynamically, NOT pre-allocated for 10M tokens
3. Activations: Temporary, released after forward pass
4. Overhead: ~2-3 GB

Result: Model loads successfully, KV cache allocated on-demand
```

#### SGLang Memory Allocation (10M context):
```
Total GPU Memory: 140 GB
mem-fraction-static: 0.65-0.80 (attempted values)
Static Memory Budget: 140 GB × mem_fraction_static

During Model Loading:
1. Model Weights: ~4 GB (distributed across 8 GPUs)
2. KV Cache Pool: PRE-ALLOCATED for 10M tokens × mem_fraction_static
   - For 10M context with FP8 E4M3:
   - Per GPU: ~7.8M tokens × 0.0234 MB/token = ~182 GB (theoretical)
   - Actual: Pre-allocation attempts to reserve this upfront
3. Activations: Reserved space for forward pass
4. CUDA Graph Buffers: Disabled (saves 4-10GB per GPU)

Result: OOM during model loading - cannot fit model weights + pre-allocated KV cache pool
```

### 3. **Critical Difference: Pre-allocation vs On-demand**

| Aspect | vLLM | SGLang |
|--------|------|--------|
| **KV Cache Allocation** | Dynamic, paged, on-demand | Static, pre-allocated pool |
| **Memory at Startup** | Model weights only (~4 GB) | Model weights + Full KV cache pool |
| **Memory Growth** | Grows with actual sequence length | Fixed at `context-length` |
| **10M Context Impact** | Only allocates what's needed | Tries to allocate full 10M capacity upfront |
| **OOM Risk** | Lower (allocates incrementally) | Higher (requires full capacity at startup) |

### 4. **Why SGLang Fails at 10M Context**

**The Problem:**
1. SGLang calculates required KV cache pool size for 10M tokens
2. With FP8 E4M3: ~7.8M tokens per GPU × 0.0234 MB/token ≈ **182 GB per GPU** (theoretical max)
3. Even with `mem-fraction-static=0.65`: 140 GB × 0.65 = **91 GB budget**
4. But SGLang tries to **pre-allocate** the KV cache pool during model loading
5. Model weights (~4 GB) + Pre-allocated KV cache pool + Activations + Overhead > 140 GB
6. **Result**: OOM during model loading phase

**Why vLLM Succeeds:**
1. vLLM does **NOT** pre-allocate KV cache for 10M tokens at startup
2. Model loads with just weights (~4 GB)
3. KV cache is allocated **on-demand** as requests come in
4. With PagedAttention, blocks are allocated incrementally
5. **Result**: Model loads successfully, KV cache grows dynamically

### 5. **Memory Allocation Code Evidence**

#### SGLang (`server_args.py:714-723`):
```python
# GPU memory capacity = model weights + KV cache pool + activations + cuda graph buffers
# mem_fraction_static = (model weights + KV cache pool) / GPU memory capacity
# Reserved memory = activations + cuda graph buffers
reserved_mem = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2
mem_fraction_static = (GPU memory capacity - reserved_mem) / GPU memory capacity
```

This shows SGLang **pre-calculates and reserves** memory for the KV cache pool at startup.

#### vLLM (from documentation):
- Uses **PagedAttention** with dynamic block allocation
- KV cache blocks are allocated **on-demand** based on actual sequence length
- No upfront pre-allocation for `max-model-len`

### 6. **Additional Factors**

#### Model Loading Phase Memory:
- **SGLang**: During model loading, it needs to:
  1. Load model weights (~4 GB)
  2. **Pre-allocate KV cache pool** (for 10M context, this is huge)
  3. Reserve space for activations
  4. Initialize memory pools
  
- **vLLM**: During model loading, it needs to:
  1. Load model weights (~4 GB)
  2. Initialize PagedAttention block manager (minimal memory)
  3. KV cache blocks allocated later on-demand

#### Memory Fragmentation:
- **SGLang**: Pre-allocation may cause fragmentation if the pool size is large
- **vLLM**: Paged allocation reduces fragmentation by using fixed-size blocks

### 7. **Why Lower `mem-fraction-static` Doesn't Help**

Even with `mem-fraction-static=0.65`:
- Budget: 140 GB × 0.65 = 91 GB
- Model weights: ~4 GB
- Available for KV cache: ~87 GB
- But SGLang still tries to **pre-allocate** KV cache pool for 10M context
- The pre-allocation calculation may still exceed available memory
- **Root issue**: Pre-allocation strategy, not just the fraction

### 8. **Solution Implications**

#### For SGLang to Support 10M Context:
1. **Change allocation strategy**: Move from static pre-allocation to dynamic on-demand allocation
2. **CPU Offload**: Offload model weights to reduce GPU memory pressure during loading
3. **Lazy KV Cache Allocation**: Allocate KV cache blocks on-demand, not upfront
4. **Reduce Context Length**: Accept limitation (e.g., 5M-6M tokens max)

#### Why vLLM Works:
- **PagedAttention**: Dynamic block allocation is the key
- **On-demand allocation**: Only allocates what's needed
- **Efficient memory use**: No wasted pre-allocated space

## Conclusion

**Root Cause**: SGLang uses **static pre-allocation** for KV cache pool, requiring full capacity upfront during model loading. vLLM uses **dynamic paged allocation**, allocating KV cache on-demand.

**Impact**: For 10M context, SGLang tries to pre-allocate ~182 GB per GPU (theoretical) during startup, causing OOM. vLLM only allocates model weights (~4 GB) at startup, then grows KV cache dynamically.

**Recommendation**: For 10M+ context length, use **vLLM** which is designed for dynamic memory allocation. SGLang would need architectural changes to support such large contexts with its current pre-allocation strategy.
