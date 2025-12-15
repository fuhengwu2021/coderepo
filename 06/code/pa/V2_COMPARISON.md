# PagedAttention v1 vs v2 对比

## 概述

我们提供了两个版本的 PagedAttention 实现：

- **v1** (`paged_attention.py` + `inference.py`): 基础版本，使用 `torch.cat` 拼接 blocks 后计算
- **v2** (`paged_attention_v2.py` + `inference_v2.py`): 使用 Online Softmax，按 block 流式处理

## 核心区别

### v1: 拼接后计算 (Concatenate-then-Compute)

```python
# 1. 收集所有 blocks
k_list = []
v_list = []
for block in blocks:
    k_list.append(block.k_cache[:num_valid])
    v_list.append(block.v_cache[:num_valid])

# 2. 拼接成连续矩阵
k_cached = torch.cat(k_list, dim=0)  # O(L) 拷贝
v_cached = torch.cat(v_list, dim=0)

# 3. 标准 attention 计算
scores = Q @ K^T
attn_weights = softmax(scores)
output = attn_weights @ V
```

**问题**：
- 每步都需要 O(L) 的拼接拷贝
- 需要分配大的连续内存（total_tokens × num_heads × head_dim）
- 无法体现"解决碎片化后不需要重排"的优势

### v2: Online Softmax (Block-Streaming)

```python
# PASS 1: 计算全局 max（遍历所有 blocks）
global_max = -inf
for block in blocks:
    scores_block = Q @ K_block^T
    block_max = max(scores_block)
    global_max = max(global_max, block_max)

# PASS 2: 使用全局 max 计算 weighted sum（遍历所有 blocks）
weighted_sum = 0
log_sum_exp = 0
for block in blocks:
    scores_block = Q @ K_block^T
    exp_scores = exp(scores_block - global_max)  # 数值稳定
    weighted_sum += sum(exp_scores * V_block)
    log_sum_exp += sum(exp_scores)

# 归一化
output = weighted_sum / log_sum_exp
```

**优势**：
- ✅ 不需要拼接 blocks（避免 O(L) 拷贝）
- ✅ 内存占用更小（不需要大连续 tensor）
- ✅ 真正的 block-streaming 计算
- ✅ 体现 PagedAttention 的核心价值

## 算法细节

### Online Softmax (Log-Sum-Exp 两段式)

Online Softmax 使用数值稳定的 log-sum-exp 算法：

1. **第一遍**：计算全局最大值 `m = max(scores)`
2. **第二遍**：使用全局 max 归一化
   - `exp_scores = exp(scores - m)`
   - `weighted_sum = sum(exp_scores * V)`
   - `log_sum_exp = sum(exp_scores)`
   - `output = weighted_sum / log_sum_exp`

这样可以避免：
- 数值溢出（通过减去全局 max）
- 大矩阵拼接（按 block 流式处理）

## 性能对比

| 特性 | v1 (Concatenate) | v2 (Online Softmax) |
|------|------------------|---------------------|
| 内存拷贝 | O(L) 每步 | O(1) 每 block |
| 中间内存 | O(L × H × D) | O(block_size × H × D) |
| 计算复杂度 | 相同 | 相同 |
| 体现 PA 优势 | ❌ | ✅ |

## 使用方式

### v1 (基础版本)
```bash
python inference.py
```

### v2 (Online Softmax)
```bash
python inference_v2.py
```

## 代码结构

```
pa/
├── paged_attention.py      # v1: 拼接后计算
├── paged_attention_v2.py   # v2: Online Softmax
├── inference.py            # v1 的推理脚本
├── inference_v2.py         # v2 的推理脚本
└── block_manager.py        # 共享的 block 管理（两个版本共用）
```

## 总结

- **v1** 适合学习和理解 PagedAttention 的基本概念（block 管理、消除 padding FLOPs）
- **v2** 展示了真正的 PagedAttention 计算形态（block-streaming、online softmax、避免拼接）

两个版本都能正确生成文本，但 v2 更接近生产系统的实现方式。
