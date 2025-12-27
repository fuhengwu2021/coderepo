# Critical Fixes Applied (Based on advice/3.md)

This document summarizes the critical fixes applied to address the "inglesingles..." output collapse issue.

## Fix 1: Attention Causal Mask (CRITICAL - NaN Prevention)

**Problem**: Using `torch.ones(...) * -inf` causes `0 * -inf = NaN`, which propagates through softmax and corrupts all activations.

**Fix**: Use boolean mask with `masked_fill`:
```python
# Before (WRONG - causes NaN):
causal_mask = torch.ones(...) * float('-inf')
scores = scores + causal_mask

# After (CORRECT):
mask = torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool).triu(1)
scores = scores.masked_fill(mask[None, None, :, :], float("-inf"))
```

**File**: `attention.py` (line ~180)

## Fix 2: QKVParallelLinear Weight Sharding Dimension (CRITICAL - Weight Semantics)

**Problem**: Weight sharding was done along `dim=1` (in_features) instead of `dim=0` (out_features). This caused:
- Wrong weight semantics (sharding input channels instead of output channels)
- Incorrect attention computation
- Model equivalent to random linear mapping

**Fix**: Shard along `dim=0` (out_features):
```python
# Before (WRONG):
return weight.narrow(1, start_idx, shard_size)  # Sharding dim=1 (in_features)

# After (CORRECT):
return weight.narrow(0, start_idx, shard_size)  # Sharding dim=0 (out_features)
```

**Note**: PyTorch `Linear.weight` has shape `[out_features, in_features]`. For ColumnParallelLinear, we shard along `dim=0` (output channels).

**Files**: 
- `linear.py` - `_shard_weight()` method (line ~504)
- `linear.py` - `weight_loader_qkv()` method (line ~458)

## Fix 3: Weight Loader Fallback (CRITICAL - LM Head Loading)

**Problem**: `weight_loader.py` used `AutoModel.from_pretrained()` as fallback, which may not include `lm_head`, causing:
- Missing or incorrect output head
- Generation always biased to certain tokens
- Output collapse

**Fix**: Use `AutoModelForCausalLM.from_pretrained()`:
```python
# Before (WRONG):
model = AutoModel.from_pretrained(...)

# After (CORRECT):
model = AutoModelForCausalLM.from_pretrained(...)
```

**File**: `weight_loader.py` (lines ~89, ~103)

## Fix 4: RoPE Application (FIXED)

**Problem**: RoPE was not being applied correctly. The calling convention was wrong.

**Fix**: Qwen2's `rotary_emb` uses:
```python
cos, sin = rotary_emb(tensor, position_ids)  # Returns (cos, sin)
q, k = apply_rotary_pos_emb(q, k, cos, sin)  # Apply RoPE
```

NOT `rotary_emb(position_ids, q, k)`.

**Implementation**:
1. Transpose q/k to [batch, seq_len, num_heads, head_dim] for cos/sin extraction
2. Call `rotary_emb(k_for_rope, position_ids)` to get cos/sin
3. Call `apply_rotary_pos_emb(q, k, cos, sin)` to apply RoPE

**Files**: 
- `attention.py` (lines ~125-149) - Fixed RoPE application logic
- `hybrid_tp_model.py` (lines ~50-62, ~217-219) - RoPE extraction from model.model.rotary_emb

## Expected Results After Fixes

After these fixes, the model should:
1. ✅ Not produce NaN values in attention scores
2. ✅ Have correct weight semantics (proper Q/K/V sharding)
3. ✅ Load complete model including `lm_head`
4. ✅ Generate coherent text instead of "inglesingles..." collapse

## Verification Steps

To verify the fixes work:

1. **Check for NaN**: Add assertions in attention forward:
   ```python
   assert not torch.isnan(scores).any(), "NaN in attention scores!"
   assert not torch.isnan(attn_weights).any(), "NaN in attention weights!"
   ```

2. **Verify weight shapes**: Check that Q/K/V shards have correct shapes:
   ```python
   # After loading, verify shard shapes
   print(f"Q shard shape: {q_shard.shape}")  # Should be [num_heads_local * head_size, hidden_size]
   ```

3. **Test with TP=1**: If TP=1 still produces wrong output, the issue is not in TP communication but in weight loading or RoPE.

4. **Compare with HF baseline**: Load same model with HF and compare intermediate activations.

## Next Steps

If output is still incorrect after these fixes:
1. Add debug assertions for NaN detection
2. Verify RoPE is actually being called (add print statements)
3. Check that position_ids are correctly generated
4. Compare weight values between TP shards and original checkpoint (using all_gather)
