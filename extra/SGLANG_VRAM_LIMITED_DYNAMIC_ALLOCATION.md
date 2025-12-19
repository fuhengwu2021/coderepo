# SGLang: VRAM-Limited Dynamic Allocation Analysis
## 为什么不能根据机器 VRAM 设置上限，然后动态分配？

## 问题分析

### 当前 SGLang 的行为

SGLang **确实会根据 VRAM 计算上限**，但问题是它仍然**预分配整个 pool**，而不是动态分配。

#### 代码证据 (`model_runner.py:1363-1444`)

```python
def profile_max_num_token(self, total_gpu_memory: int):
    # 计算每个 token 的 KV cache 大小
    cell_size = ...  # 根据模型配置计算
    
    # 计算可用内存
    rest_memory = available_gpu_memory - total_gpu_memory * (
        1 - self.mem_fraction_static
    )
    
    # 根据可用内存计算最大 token 数
    max_num_token = int(rest_memory * (1 << 30) // cell_size)
    return max_num_token
```

**关键问题**：
1. ✅ SGLang **确实计算**了基于 VRAM 的最大 token 数
2. ❌ 但它会**预分配整个 pool**（在 `init_memory_pool` 中）
3. ❌ 即使设置了 `context-length=10000000`，它也会尝试预分配对应的 pool

### 为什么预分配会导致 OOM？

#### 场景：10M Context on 8x H200

```
用户设置: --context-length 10000000
GPU Memory: 140 GB per GPU
mem-fraction-static: 0.65

SGLang 的计算过程：
1. 计算可用内存: 140 GB × 0.65 = 91 GB
2. 计算每个 token 的 KV cache: ~0.0234 MB/token (FP8 E4M3)
3. 计算最大 token 数: 91 GB ÷ 0.0234 MB/token ≈ 3.9M tokens per GPU
4. 但是用户设置了 context-length=10M，SGLang 会尝试预分配 10M tokens 的 pool
5. 10M tokens × 0.0234 MB/token ≈ 234 GB per GPU (超过 140 GB)
6. 结果: OOM at startup
```

**根本问题**：
- SGLang 使用 `context-length` 作为**预分配大小**，而不是**最大限制**
- 即使计算出的 `max_num_token` 小于 `context-length`，它仍然会尝试预分配 `context-length` 的 pool

---

## 解决方案：VRAM-Limited Dynamic Allocation

### 方案 1: 基于 VRAM 上限的动态分配（推荐）

#### 核心思想
1. **根据 VRAM 计算实际可用的最大 token 数**
2. **预分配一个较小的 pool**（例如 1M tokens）
3. **动态扩展**到 VRAM 上限，而不是 `context-length`

#### 实现方式

```python
# 伪代码
def init_memory_pool_vram_limited(self, total_gpu_memory):
    # 1. 计算基于 VRAM 的实际最大 token 数
    vram_max_tokens = self.profile_max_num_token(total_gpu_memory)
    
    # 2. 使用较小的初始 pool（例如 1M tokens 或 vram_max_tokens 的 10%）
    initial_pool_size = min(1_000_000, vram_max_tokens // 10)
    
    # 3. 预分配初始 pool
    self.kv_cache_pool = pre_allocate_pool(initial_pool_size)
    
    # 4. 设置动态扩展上限为 VRAM 计算的最大值
    self.max_pool_size = vram_max_tokens
    
    # 5. 实现动态扩展逻辑
    def expand_pool_if_needed(current_size, requested_size):
        if requested_size > current_size and current_size < self.max_pool_size:
            # 动态扩展 pool
            new_size = min(requested_size * 1.5, self.max_pool_size)
            expand_pool(new_size)
```

#### 优点
- ✅ **避免 OOM**: 只预分配小 pool，启动成功
- ✅ **支持大 context**: 可以动态扩展到 VRAM 上限
- ✅ **内存效率**: 只分配实际使用的内存
- ✅ **保持性能**: 初始 pool 足够大，减少频繁扩展

#### 缺点
- ❌ **实现复杂度**: 需要实现动态扩展逻辑
- ❌ **扩展延迟**: Pool 扩展可能有延迟（但可以预扩展）
- ❌ **碎片化风险**: 动态扩展可能导致内存碎片

---

### 方案 2: 限制 context-length 到 VRAM 上限

#### 核心思想
**自动将 `context-length` 限制到 VRAM 计算的最大值**

#### 实现方式

```python
# 在 server_args.py 中
def adjust_context_length_by_vram(self):
    # 计算基于 VRAM 的最大 token 数
    vram_max_tokens = self.calculate_vram_max_tokens()
    
    # 如果用户设置的 context-length 超过 VRAM 限制，自动调整
    if self.context_length > vram_max_tokens:
        logger.warning(
            f"context-length ({self.context_length}) exceeds VRAM limit "
            f"({vram_max_tokens}). Adjusting to {vram_max_tokens}."
        )
        self.context_length = vram_max_tokens
```

#### 优点
- ✅ **简单实现**: 只需在启动时调整参数
- ✅ **避免 OOM**: 确保不会超过 VRAM 限制
- ✅ **向后兼容**: 不需要改变现有架构

#### 缺点
- ❌ **功能限制**: 用户无法使用超过 VRAM 限制的 context
- ❌ **仍然预分配**: 仍然预分配整个 pool（虽然更小）
- ❌ **不够灵活**: 无法充分利用可用内存

---

### 方案 3: 混合策略（最佳）

#### 核心思想
**结合方案 1 和方案 2**：
1. 自动限制 `context-length` 到 VRAM 上限
2. 预分配较小的初始 pool
3. 支持动态扩展到 VRAM 上限

#### 实现方式

```python
def init_memory_pool_hybrid(self, total_gpu_memory):
    # 1. 计算 VRAM 上限
    vram_max_tokens = self.profile_max_num_token(total_gpu_memory)
    
    # 2. 限制 context-length 到 VRAM 上限
    if self.context_length > vram_max_tokens:
        logger.warning(f"Limiting context-length from {self.context_length} to {vram_max_tokens}")
        self.context_length = vram_max_tokens
    
    # 3. 预分配较小的初始 pool（例如 50% 或 1M tokens，取较小值）
    initial_pool_size = min(
        max(1_000_000, vram_max_tokens // 2),  # 至少 1M，最多 50%
        vram_max_tokens
    )
    
    # 4. 预分配初始 pool
    self.kv_cache_pool = pre_allocate_pool(initial_pool_size)
    
    # 5. 设置动态扩展上限
    self.max_pool_size = vram_max_tokens
    
    # 6. 实现按需扩展
    self.enable_dynamic_expansion = True
```

#### 优点
- ✅ **避免 OOM**: 启动时只分配小 pool
- ✅ **支持大 context**: 可以扩展到 VRAM 上限
- ✅ **自动限制**: 防止用户设置过大的 context-length
- ✅ **性能平衡**: 初始 pool 足够大，减少扩展频率

#### 缺点
- ❌ **实现复杂度**: 需要实现动态扩展
- ❌ **需要测试**: 更多边界情况需要测试

---

## 为什么当前实现不这样做？

### 技术原因

1. **架构设计**: SGLang 的 Radix Cache 和 Prefix Caching 依赖于**固定大小的预分配 pool**
2. **性能优化**: 预分配避免了运行时分配开销，提高了性能
3. **简化实现**: 静态分配使代码更简单，更容易调试

### 历史原因

- SGLang 最初设计用于**中小型 context**（< 1M tokens）
- 对于这些场景，预分配是**最优策略**
- 大 context（10M+）是**新需求**，需要架构调整

---

## 实施建议

### 短期方案（立即可行）

**自动限制 context-length 到 VRAM 上限**：

```python
# 在 run-sglang-docker.sh 或 server_args.py 中
# 计算基于 VRAM 的最大 context length
VRAM_GB = 140  # H200
mem_fraction = 0.65
available_memory = VRAM_GB * mem_fraction  # ~91 GB
model_weights = 4  # GB
kv_cache_memory = available_memory - model_weights  # ~87 GB
bytes_per_token = 0.0234 * 1024 * 1024  # FP8 E4M3
max_tokens = int(kv_cache_memory * 1024**3 / bytes_per_token)  # ~3.9M tokens

# 如果用户设置的 context-length 超过这个值，自动调整
if context_length > max_tokens:
    context_length = max_tokens
    logger.warning(f"Adjusted context-length to {max_tokens} based on VRAM limit")
```

**优点**：
- ✅ 简单实现，只需几行代码
- ✅ 立即解决 OOM 问题
- ✅ 不需要架构变更

**缺点**：
- ❌ 仍然预分配整个 pool
- ❌ 无法充分利用可用内存（如果实际使用 < max_tokens）

### 中期方案（6-12 个月）

**实现动态扩展**：
1. 预分配较小的初始 pool（例如 1M tokens）
2. 实现按需扩展逻辑
3. 支持扩展到 VRAM 上限

### 长期方案（12+ 个月）

**完全动态分配**（类似 vLLM）：
1. 实现 PagedAttention 风格的块分配
2. 完全按需分配，无预分配
3. 最大化内存效率

---

## 结论

### 回答用户问题

**Q: 为什么不能根据机器 VRAM 设置上限，然后动态分配？**

**A: 技术上可以，但需要架构调整**

1. **当前状态**: SGLang 会根据 VRAM 计算上限，但**仍然预分配整个 pool**
2. **问题根源**: `context-length` 被用作**预分配大小**，而不是**最大限制**
3. **解决方案**: 
   - **短期**: 自动限制 `context-length` 到 VRAM 上限（简单）
   - **中期**: 实现动态扩展（需要开发）
   - **长期**: 完全动态分配（类似 vLLM，需要重大重构）

### 推荐方案

**立即实施**: 自动限制 `context-length` 到 VRAM 上限
- 简单、有效、无需架构变更
- 可以立即解决 10M context 的 OOM 问题

**未来规划**: 实现动态扩展
- 提供更好的内存效率
- 支持更灵活的使用场景
