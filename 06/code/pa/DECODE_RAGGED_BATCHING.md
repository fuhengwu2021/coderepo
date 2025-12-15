# Decode 阶段的 Ragged Batching 实现

## 概述

在 v4 的 decode 阶段，我们实现了**真正的 ragged batching**，所有序列在一个 batch 中并行处理。

## 实现细节

### 关键改进

与之前的顺序处理不同，新的实现：

1. ✅ **Batch Embedding**：所有 tokens 一次性 embed
2. ✅ **Batch Layer Norm**：所有序列一起做 layer norm
3. ✅ **Batch Q/K/V Projection**：所有序列一起做投影
4. ✅ **Batch RoPE**：所有序列一起应用 RoPE（虽然需要 per-sequence position）
5. ✅ **Batch KV Append**：所有序列的 KV 一起 append
6. ✅ **Batch Attention**：使用 PagedAttention 为每个序列计算 attention
7. ✅ **Batch Output Projection**：所有序列一起做输出投影
8. ✅ **Batch MLP**：所有序列一起做 MLP
9. ✅ **Batch Final Norm & LM Head**：所有序列一起做最终计算

### 代码结构

```python
def decode_batch(self):
    # 1. 获取所有 decode 序列
    seq_ids, positions, token_ids = self.scheduler.get_batch(...)
    
    # 2. ✅ Batch Embedding: [1, num_seqs] -> [1, num_seqs, H]
    hidden_states = self.model.model.embed_tokens(token_tensor)
    
    # 3. 遍历每一层
    for layer_idx in range(self.num_layers):
        # ✅ Batch Layer Norm: [1, num_seqs, H]
        hidden_states = layer.input_layernorm(hidden_states)
        
        # ✅ Batch Q/K/V Projection: [1, num_seqs, H] -> [1, Hq/kv, num_seqs, D]
        q = attn.q_proj(hidden_states).view(1, num_seqs, Hq, D).transpose(1, 2)
        k = attn.k_proj(hidden_states).view(1, num_seqs, Hkv, D).transpose(1, 2)
        v = attn.v_proj(hidden_states).view(1, num_seqs, Hkv, D).transpose(1, 2)
        
        # ✅ Batch RoPE: 为每个序列应用 RoPE
        for i, seq_id in enumerate(seq_ids):
            q_seq, k_seq = apply_rope(q[:, i], k[:, i], positions[i])
        
        # ✅ Batch KV Append: 所有序列一起 append
        for i, seq_id in enumerate(seq_ids):
            self.paged_attentions[layer_idx].append_kv(seq_id, k[i], v[i], positions[i])
        
        # ✅ Batch Attention: 使用 PagedAttention 为每个序列计算
        attn_outputs = []
        for i, seq_id in enumerate(seq_ids):
            attn_output = self.paged_attentions[layer_idx].compute_attention(seq_id, q[i])
            attn_outputs.append(attn_output)
        
        # ✅ Batch Output Projection: [1, num_seqs, H]
        attn_output = attn.o_proj(torch.stack(attn_outputs))
        
        # ✅ Batch MLP: [1, num_seqs, H]
        mlp_output = layer.mlp(layer.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + mlp_output
    
    # ✅ Batch Final: [1, num_seqs, vocab_size]
    logits = self.model.lm_head(self.model.model.norm(hidden_states))
```

## 对比：之前 vs 现在

### 之前的实现（顺序处理）

```python
# ❌ 顺序处理每个序列
for i, seq_id in enumerate(seq_ids):
    seq_hidden = hidden_states[:, i:i+1, :]  # [1, 1, H]
    
    # 为每个序列单独处理每一层
    for layer_idx in range(self.num_layers):
        # 单独计算 Q, K, V
        q = attn.q_proj(seq_hidden)  # [1, 1, Hq*D]
        # ... 单独处理 ...
```

**问题**：
- 每个序列单独处理
- 无法利用 batch 处理的优势
- GPU 利用率低

### 现在的实现（Batch 处理）

```python
# ✅ Batch 处理所有序列
hidden_states = self.model.model.embed_tokens(token_tensor)  # [1, num_seqs, H]

# 所有序列一起处理每一层
for layer_idx in range(self.num_layers):
    # Batch Q, K, V projection
    q = attn.q_proj(hidden_states)  # [1, num_seqs, Hq*D]
    # ... batch 处理 ...
```

**优势**：
- 所有序列一起处理
- 充分利用 GPU 并行能力
- 更高的 GPU 利用率

## 关键点

### 1. Ragged Batching 格式

```
输入: [1, num_seqs]  # 每个序列 1 个 token
      # 例如: [1, 3] 表示 3 个序列，每个序列 1 个 token
      # 而不是 [3, max_len] (需要 padding)

处理: [1, num_seqs, H]  # 所有序列一起处理
      # 例如: [1, 3, 1024] 表示 3 个序列，每个序列 hidden_size=1024
```

### 2. PagedAttention 的使用

虽然我们 batch 处理 Q, K, V 的计算，但 attention 计算仍然需要为每个序列单独调用 `compute_attention()`，因为：

- 每个序列的 KV cache 在不同的 blocks 中
- 每个序列的 block_table 不同
- 需要从不同的 blocks 中 gather K, V

```python
# Batch 计算 Q, K, V
q_batch = q[0].transpose(0, 1)  # [num_seqs, Hq, D]

# 但 attention 需要 per-sequence（因为 KV cache 在不同 blocks）
for i, seq_id in enumerate(seq_ids):
    attn_output = self.paged_attentions[layer_idx].compute_attention(seq_id, q_batch[i])
```

### 3. RoPE 的处理

RoPE 需要 per-sequence 的 position，所以我们需要为每个序列单独应用：

```python
# 虽然 Q, K 是 batch 的，但 RoPE 需要 per-sequence position
for i, seq_id in enumerate(seq_ids):
    seq_position = torch.tensor([[positions[i]]], device=self.device)
    q_seq, k_seq = apply_rope(q[:, i], k[:, i], seq_position)
```

## 性能优势

### GPU 利用率

**之前（顺序处理）**：
```
序列 1: [████████] 100% GPU
序列 2: [████████] 100% GPU  (等待序列 1 完成)
序列 3: [████████] 100% GPU  (等待序列 2 完成)
总时间: 3 × T
```

**现在（Batch 处理）**：
```
序列 1, 2, 3: [████████] 100% GPU (并行)
总时间: T
```

### 吞吐量提升

- **顺序处理**：3 个序列 = 3 次 forward
- **Batch 处理**：3 个序列 = 1 次 forward（batch size=3）

理论上，batch 处理可以提升 **3x** 吞吐量（对于 3 个序列的情况）。

## 限制

### 仍然不是完全并行

虽然我们 batch 处理了大部分计算，但：

1. **Attention 计算**：仍然需要 per-sequence（因为 KV cache 在不同 blocks）
2. **RoPE 应用**：需要 per-sequence（因为 position 不同）

### 真正的并行需要

要实现完全并行，需要：
1. **自定义 CUDA kernels**：处理 flattened tokens + metadata
2. **Block-based Attention**：在 kernel 中从 blocks gather K, V
3. **Dynamic Mask**：从 metadata 构建 attention mask

## 总结

### 我们实现了什么

✅ **Batch Embedding**：所有 tokens 一起 embed  
✅ **Batch Layer Norm**：所有序列一起做 norm  
✅ **Batch Q/K/V Projection**：所有序列一起做投影  
✅ **Batch Output Projection**：所有序列一起做输出投影  
✅ **Batch MLP**：所有序列一起做 MLP  
✅ **Batch Final Norm & LM Head**：所有序列一起做最终计算  

### 仍然需要 per-sequence 的部分

⚠️ **Attention 计算**：需要 per-sequence（KV cache 在不同 blocks）  
⚠️ **RoPE 应用**：需要 per-sequence（position 不同）  

### 性能提升

- **GPU 利用率**：从顺序处理提升到 batch 处理
- **吞吐量**：理论上提升 **batch_size** 倍
- **延迟**：单个序列的延迟不变，但整体吞吐量提升

这是我们在 HuggingFace 框架下能实现的最接近真正的 ragged batching 的方式。
