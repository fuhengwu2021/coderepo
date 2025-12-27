# Bias Loading Fix

## Problem
Qwen2.5-0.5B-Instruct has `bias=True` for q_proj, k_proj, and v_proj, but our TP implementation was creating layers with `bias=False`.

## Fix
1. **Detect bias from state_dict**: Check if bias keys exist in state_dict
2. **Set bias=True when creating layers**: Pass `bias=has_bias` to TensorParallelAttention
3. **Load bias weights**: Added `bias_loader_qkv` method to QKVParallelLinear to load and shard Q/K/V biases
4. **Load biases separately**: For separate Q/K/V loading, call `bias_loader_qkv` for each

## Files Changed
- `linear.py`: Added `bias_loader_qkv` method to QKVParallelLinear
- `model_wrapper.py`: 
  - Detect bias from state_dict
  - Set `bias=has_bias` when creating TensorParallelAttention
  - Load Q/K/V biases using `bias_loader_qkv`

## Results
After fix:
- Q projection: ✅ Matches (diff=0.000000)
- K projection: ✅ Almost matches (diff=0.000015, likely numerical precision)
- V projection: ✅ Matches (diff=0.000000)
- Final logits: Improved from 23.31 max diff to 15.34 max diff (still needs more fixes)
