# RoPE Application Fix

## Problem
The output was still incorrect (garbled text) even after fixing weight sharding and causal mask.

## Root Cause
The RoPE (Rotary Position Embedding) was not being applied correctly. The calling convention was wrong.

## Fix
Qwen2's `rotary_emb` uses:
```python
cos, sin = rotary_emb(tensor, position_ids)  # Returns (cos, sin)
q, k = apply_rotary_pos_emb(q, k, cos, sin)  # Apply RoPE
```

NOT:
```python
q, k = rotary_emb(position_ids, q, k)  # This doesn't work for Qwen2
```

## Implementation
1. Transpose q/k to [batch, seq_len, num_heads, head_dim] for cos/sin extraction
2. Call `rotary_emb(k_for_rope, position_ids)` to get cos/sin
3. Call `apply_rotary_pos_emb(q, k, cos, sin)` to apply RoPE
4. q/k remain in [batch, num_heads, seq_len, head_dim] format for attention

## Files Changed
- `attention.py`: Fixed RoPE application logic (lines ~125-149)
