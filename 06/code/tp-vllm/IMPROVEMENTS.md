# Improvements Based on Advice

This document summarizes the critical fixes applied based on the advice in `advice/2.md`.

## Critical Fixes Applied

### A. Fixed KV Cache Return for Incremental Decode ✅

**Problem**: After `repeat_interleave` for GQA, the code returned repeated heads as cache, causing memory explosion.

**Fix**: Keep separate `k_kv`/`v_kv` for storage (num_kv_heads_local) and only repeat for attention computation.

**Changes in `attention.py`**:
- Store `k_kv`, `v_kv` with original num_kv_heads_local
- Create `k_for_attn`, `v_for_attn` with repeated heads only for attention
- Return `(k_kv, v_kv)` as cache, not the repeated version

**Result**: KV cache now correctly stores only KV heads, preventing memory growth.

### B. Fixed Causal Mask NaN Issue ✅

**Problem**: Using `triu(ones) * -inf` produces NaN because `0 * -inf = NaN` in PyTorch.

**Fix**: Use boolean mask with `masked_fill` instead.

**Changes in `attention.py`**:
```python
# Before (produces NaN):
causal_mask = torch.triu(torch.ones(...), diagonal=1) * float('-inf')
scores = scores + causal_mask

# After (correct):
causal_mask = torch.triu(torch.ones(..., dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
```

**Result**: No more NaN values in attention scores.

### C. Fixed QKVParallelLinear Sizing Logic ✅

**Problem**: Mixed global/local sizing with pre-multiplication hack that was confusing and fragile.

**Fix**: Use global output_size (total across all ranks) and let ColumnParallelLinear divide internally. Added assertion to verify correctness.

**Changes in `linear.py`**:
- Changed from: `output_size = (num_heads_local + 2*num_kv_heads_local) * tp_size * head_size`
- Changed to: `global_output_size = (total_num_heads + 2*total_num_kv_heads) * head_size`
- Added assertion: `assert self.weight.shape[0] == expected_local_output_size`

**Result**: Clearer, more maintainable code that matches vLLM's approach.

### D. Improved Weight Loading ✅

**Problem**: Full state_dict loaded on every rank, wasting memory.

**Fix**: Added `load_on_all_ranks` parameter. When False, only rank0 loads (more memory-efficient). Added TODO for future optimization to load shards directly from safetensors.

**Changes in `weight_loader.py`**:
- Added `load_on_all_ranks` parameter (default True for backward compatibility)
- When False, only rank0 loads full weights
- Added comments about future optimization opportunities

**Result**: Foundation for more memory-efficient weight loading (can be extended later).

### E. Moved inference_mode to Top Level ✅

**Problem**: `inference_mode` inside layer forward methods can be awkward and surprising.

**Fix**: Removed `inference_mode` from layer forward methods. Applied at top level in demo.

**Changes**:
- Removed `with torch.inference_mode()` from `ColumnParallelLinear.forward()` and `RowParallelLinear.forward()`
- Kept `with torch.inference_mode()` in `demo_vllm.py` at the top level

**Result**: Better ergonomics, no nested inference_mode.

## Verification

All fixes have been tested:
- ✅ KV cache shapes are correct: `k=torch.Size([2, 1, 10, 64])` (num_kv_heads_local=1, not repeated)
- ✅ No NaN errors in attention scores
- ✅ QKVParallelLinear assertion passes
- ✅ Forward pass completes successfully

## Remaining Items (Not Critical)

The following items from the advice are noted but not yet implemented (they're enhancements, not correctness fixes):

1. **PagedAttention Integration**: Replace simple KV cache with block-based KV store (BlockManager + slot_mapping). This is a larger architectural change.

2. **Direct Shard Loading from Safetensors**: Each rank could load only its shard directly from safetensors files, avoiding full tensor loading even on rank0. This requires more complex implementation.

## Summary

All critical correctness issues have been fixed:
- ✅ KV cache return (prevents memory explosion)
- ✅ Causal mask NaN (prevents numerical errors)
- ✅ QKVParallelLinear sizing (clearer, more maintainable)
- ✅ Weight loading foundation (ready for optimization)
- ✅ inference_mode placement (better ergonomics)

The code is now more robust and closer to vLLM's production implementation.
