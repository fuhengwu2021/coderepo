# Fixes Based on advice/5.md

## Fix 1: Cache/Position Alignment Assertions (HIGH PRIORITY)

**Problem**: Cache length must strictly equal position_ids value, otherwise attention will see wrong positions during incremental decode.

**Fix**: Added assertions to verify cache/position alignment:
- Before append: `assert k_cache.shape[2] == position_ids.item()`
- After append: `assert k_kv.shape[2] == position_ids.item() + 1`
- In generate: Verify position matches cache length before each decode step

**Files**:
- `attention.py` (lines ~148-175): Cache alignment checks in forward
- `hybrid_tp_model.py` (lines ~410-418): Position/cache verification in generate

## Fix 2: Remove RoPE Fallback (RECOMMENDED)

**Problem**: `HybridTPDecoderLayer._apply_rope` has try/except fallback that could mask errors.

**Fix**: Removed `_apply_rope` method entirely. RoPE is now handled directly in `TensorParallelAttention` with proper error handling (no try/except).

**File**: `hybrid_tp_model.py` (removed `_apply_rope` method)

## Fix 3: Weight Loading Shape Assertions (MEDIUM PRIORITY)

**Problem**: No verification that loaded weights match expected format, could silently fail with different checkpoint formats.

**Fix**: Added hard assertions to verify:
- Fused QKV: `loaded_weight.shape[1] == hidden_size` and `loaded_weight.shape[0] == expected_total_out`
- Separate Q/K/V: Each weight must match expected `[out_features, in_features]` format

**File**: `linear.py` (weight_loader_qkv method, lines ~478-500)

## Fix 4: Golden Check Test Script (RECOMMENDED)

**Purpose**: Verify that HybridTPModel (tp=1) produces same results as HuggingFace original.

**Implementation**: Created `test_golden_check.py` that:
- Compares final logits between TP model (tp=1) and HF model
- Reports max/mean differences
- Helps identify RoPE shape or weight loading issues

**Usage**: `python test_golden_check.py`

## Summary

These fixes address the critical issues identified in advice/5.md:
1. ✅ Cache/position alignment verification (prevents decode position errors)
2. ✅ Removed RoPE fallback (prevents silent errors)
3. ✅ Weight shape assertions (prevents silent loading errors)
4. ✅ Golden check test (helps verify correctness)

The golden check is the most important - if tp=1 matches HF, then the implementation is correct. If not, it's likely a RoPE shape or weight mapping issue.
