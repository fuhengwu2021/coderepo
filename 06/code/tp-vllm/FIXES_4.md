# Fixes Based on advice/4.md

## Fix 1: RoPE Application (CRITICAL)

**Problem**: RoPE shape/调用方式可能仍然不对，try/except 吞异常导致问题难以定位。

**Fix**:
- 移除 try/except，直接对齐 HuggingFace 的调用路径
- 确保 shape 完全匹配：
  - `rotary_emb` 期望 `[batch, seq_len, num_heads, head_dim]`，返回 `(cos, sin)` 形状为 `[batch, seq_len, head_dim]`
  - `apply_rotary_pos_emb` 期望 `q, k` 为 `[batch, num_heads, seq_len, head_dim]`，cos/sin 会自动 broadcast

**Implementation**:
```python
# Transpose to HF format for cos/sin extraction
q_for_rope = q.transpose(1, 2)  # [batch, seq_len, num_heads_local, head_dim]
k_for_rope = k.transpose(1, 2)  # [batch, seq_len, num_kv_heads_local, head_dim]

# Get cos, sin (HF format: [batch, seq_len, head_dim])
cos, sin = rotary_emb(k_for_rope, position_ids)

# Apply RoPE (q, k are [batch, num_heads, seq_len, head_dim])
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

**File**: `attention.py` (lines ~125-142)

## Fix 2: KV Cache Incremental Decode (HIGH PRIORITY)

**Problem**: `generate` 方法没有真正使用 KV cache，每一步都在做 full forward。

**Fix**:
- 修改 `forward` 返回 `(logits, kv_caches)`
- Prefill: `forward(input_ids, kv_caches=None)` → 返回 kv_caches
- Decode: 每步只 forward 新 token，传入并更新 kv_caches
- Position IDs: 使用绝对位置（当前序列长度）

**Implementation**:
```python
# Prefill
logits, kv_caches = self.forward(input_ids, kv_caches=None)

# Decode loop
for step in range(max_new_tokens):
    # Sample next token
    next_token = sample(next_token_logits)
    
    # Incremental decode: only forward new token
    new_token_ids = next_token.unsqueeze(0)  # [batch, 1]
    current_position = generated_ids.shape[1] - 1
    position_ids = torch.tensor([[current_position]], device=self.device)
    
    logits, kv_caches = self.forward(new_token_ids, position_ids=position_ids, kv_caches=kv_caches)
    next_token_logits = logits[0, -1, :]
```

**Files**: 
- `hybrid_tp_model.py` - `forward()` 方法 (returns kv_caches)
- `hybrid_tp_model.py` - `generate()` 方法 (incremental decode)

## Fix 3: GQA Assertion (MEDIUM PRIORITY)

**Problem**: 没有明确验证 GQA 的 head 对齐规则。

**Fix**: 添加断言确保 `num_heads_local % num_kv_heads_local == 0`

**File**: `linear.py` (QKVParallelLinear.__init__)

## Summary

按照 advice/4.md 的建议，优先修复了：
1. ✅ RoPE: 移除 try/except，严格对齐 HF 调用路径
2. ✅ KV Cache: 实现真正的增量 decode
3. ✅ GQA: 添加 head 对齐断言

这些修复应该显著改善生成质量和性能。
