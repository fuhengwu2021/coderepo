# Prefill Batching 解释

## 什么是 Prefill Batching？

**Prefill Batching** 指的是：在同一个 batch 中同时处理多个序列的 prefill 阶段（处理 prompt tokens），而不是每个序列单独 prefill。

## 当前实现（v3）的做法

### 当前方式：立即 Prefill

```python
def add_request(self, prompt: str, max_new_tokens: int = 50):
    # 1. Tokenize prompt
    prompt_tokens = tokenize(prompt)
    
    # 2. 立即单独 prefill（不等待其他请求）
    self._prefill_sequence(seq_id, prompt_tokens)  # ← 单独处理
    
    # 3. 立即转为 decode 状态
    seq_info.state = SequenceState.DECODE
```

**问题**：
- 每个请求到达时立即 prefill
- 如果有 10 个请求同时到达，会做 10 次单独的 prefill
- 无法利用 batch 处理的效率优势

## 真正的 Prefill Batching 应该怎么做

### vLLM 的方式：批量 Prefill

```python
# 1. 收集多个需要 prefill 的请求
prefill_requests = []
for new_request in incoming_requests:
    prefill_requests.append(new_request)

# 2. 在同一个 batch 中一起 prefill
# 使用 ragged batching：摊平所有 prompt tokens
all_prompt_tokens = flatten([req.prompt_tokens for req in prefill_requests])
# 例如：3 个请求，prompt 长度 [10, 15, 8] → 总共 33 个 tokens

# 3. 使用 metadata 标识每个 token
seq_id_flat = [0,0,0,..., 1,1,1,..., 2,2,...]  # 33 个元素
position_flat = [0,1,2,...,9, 0,1,2,...,14, 0,1,...,7]  # 每个序列内的位置

# 4. 一次 forward 处理所有 tokens
outputs = model.forward(
    input_ids=all_prompt_tokens,  # [33] 而不是 [3, 15] (需要 padding)
    seq_ids=seq_id_flat,
    positions=position_flat,
    ...
)
```

## 对比

### 当前实现（v3）

```
Request 1 到达 → 立即 prefill (10 tokens) → decode
Request 2 到达 → 立即 prefill (15 tokens) → decode  
Request 3 到达 → 立即 prefill (8 tokens) → decode

总共：3 次独立的 prefill，无法 batch
```

### Prefill Batching（vLLM 方式）

```
Request 1 到达 → 加入 prefill queue
Request 2 到达 → 加入 prefill queue
Request 3 到达 → 加入 prefill queue

Scheduler 决定：batch_size=3，一起 prefill
→ 一次 forward 处理 33 个 tokens（10+15+8）
→ 使用 ragged batching，无需 padding

总共：1 次 batch prefill，效率更高
```

## 为什么 Prefill Batching 重要？

1. **效率提升**：
   - 单独 prefill：10 个请求 = 10 次 forward
   - Batch prefill：10 个请求 = 1 次 forward（处理所有 tokens）

2. **GPU 利用率**：
   - Batch 处理能更好地利用 GPU 并行能力
   - 单独处理会导致 GPU 利用率低

3. **延迟 vs 吞吐量权衡**：
   - 立即 prefill：低延迟（请求到达即处理）
   - Batch prefill：高吞吐量（等待多个请求一起处理）

## 如何实现 Prefill Batching？

### 1. 修改 Scheduler

```python
class ContinuousBatchScheduler:
    def __init__(self):
        self.prefill_queue = []  # 等待 prefill 的请求
        self.decode_sequences = []  # 正在 decode 的序列
    
    def add_request(self, prompt_tokens):
        # 不立即 prefill，而是加入队列
        self.prefill_queue.append({
            'prompt_tokens': prompt_tokens,
            'seq_id': self.next_seq_id,
            ...
        })
    
    def get_prefill_batch(self, max_batch_size=32):
        # 返回一批需要 prefill 的请求
        batch = self.prefill_queue[:max_batch_size]
        self.prefill_queue = self.prefill_queue[max_batch_size:]
        return batch
```

### 2. 批量 Prefill

```python
def prefill_batch(self, prefill_requests):
    # 1. 摊平所有 prompt tokens
    all_tokens = []
    seq_ids = []
    positions = []
    
    for req in prefill_requests:
        for pos, token in enumerate(req.prompt_tokens):
            all_tokens.append(token)
            seq_ids.append(req.seq_id)
            positions.append(pos)
    
    # 2. 一次 forward（使用 ragged batching）
    tokens_tensor = torch.tensor([all_tokens], device=self.device)  # [1, T]
    # 使用 metadata 标识每个 token 属于哪个序列
    
    outputs = model.forward(...)
    
    # 3. 为每个序列缓存 KV
    for req in prefill_requests:
        cache_kv_for_sequence(req.seq_id, ...)
```

## 当前 v3 的限制

- ❌ **没有 prefill batching**：每个请求立即单独 prefill
- ❌ **没有 prefill queue**：无法收集多个请求一起处理
- ✅ **有 decode batching**：多个序列可以在 decode 阶段一起处理（虽然是顺序的）

## 总结

**Prefill Batching** = 在同一个 batch 中同时处理多个序列的 prompt tokens，使用 ragged batching（摊平 tokens + metadata）而不是单独处理每个序列。

当前 v3 实现的是：
- ✅ 序列调度和管理
- ✅ 共享 block pool
- ❌ Prefill batching（每个请求立即单独 prefill）
- ❌ 真正的 ragged batching（使用 Python loops 而不是 flattened tokens）
