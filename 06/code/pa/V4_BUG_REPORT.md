# v4 Decode Ragged Batching Bug 报告

## 问题描述

v4 的 decode 阶段实现了 batch 处理，但生成结果出现了问题：
- Request 1: "The capital of France's capital of France's..." (重复)
- Request 2: "Quantum is a simple machines that can be..." (重复)
- Request 3: "In this, here's" (重复)

而 v3 的结果是正确的。

## 根本原因

**Bug**: RoPE 应用时的索引错误

在原来的代码中：
```python
# ❌ 错误：序列维度在索引 2，不是索引 1
q_seq = q[:, i:i+1, :, :]  # 错误！这会选择 head i，而不是 sequence i
```

正确的代码：
```python
# ✅ 正确：在 transpose(1, 2) 之后，序列维度在索引 2
q_seq = q[:, :, i:i+1, :]  # 正确！选择 sequence i
```

### 详细说明

1. **Q, K, V 的 Reshape**：
   ```python
   q = q.view(1, num_seqs, self.num_heads, self.head_dim).transpose(1, 2)
   # 结果：q shape = [1, Hq, num_seqs, D]
   # 序列维度在索引 2，不是索引 1
   ```

2. **错误的 RoPE 应用**：
   ```python
   q_seq = q[:, i:i+1, :, :]  # ❌ 选择了 head i，而不是 sequence i
   ```

3. **正确的 RoPE 应用**：
   ```python
   q_seq = q[:, :, i:i+1, :]  # ✅ 选择了 sequence i
   ```

## 修复

修复后的代码：
```python
# After transpose, q shape is [1, Hq, num_seqs, D]
# So q[:, :, i, :] is sequence i
for i, seq_id in enumerate(seq_ids):
    q_seq = q[:, :, i:i+1, :]  # [1, Hq, 1, D] - sequence i
    k_seq = k[:, :, i:i+1, :]  # [1, Hkv, 1, D] - sequence i
    # Apply RoPE...
    q[:, :, i:i+1, :] = q_seq
    k[:, :, i:i+1, :] = k_seq
```

## 验证

修复后的结果：
- ✅ Request 1: "The capital of France is Paris."
- ✅ Request 2: "Quantum computing is a type of computing..."
- ✅ Request 3: "Infinite minds, Code and algorithms dance, AI whispers, unseen."

## 当前状态

- ✅ 代码可以运行
- ✅ 生成结果正确
- ✅ Bug 已修复
