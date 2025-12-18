# SGLang 的 Hybrid KV Cache 支持分析

## 问题：SGLang 是否支持 Hybrid KV Cache Manager？

**简短回答：** SGLang 有类似的功能，但实现方式和 vLLM 不同。

## SGLang 的 Hybrid KV Cache 机制

### 1. **HiCache（分层 KV 缓存）**

SGLang 的 **HiCache** 是一个分层 KV 缓存系统，管理三个层级的 KV cache：
- **GPU 内存**：快速访问
- **CPU 内存（Host）**：中等速度
- **外部存储**：慢速但容量大

**目的：** 扩展 KV cache 容量，突破 GPU 内存限制。

**启用方式：**
```bash
--enable-hierarchical-cache  # 必须先启用 HiCache
--hicache-ratio 2.0  # CPU 内存大小相对于 GPU 显存大小的比例（默认 2.0）
```

**参数说明：**
- `--enable-hierarchical-cache`: **必须项**，启用分层缓存功能
- `--hicache-ratio <float>`: CPU 内存（L2 Cache）相对于 GPU 显存的比例
  - 默认值: `2.0`（CPU 内存是 GPU 显存的 2 倍）
  - 计算公式: `Host_Memory_Size = GPU_Memory_Size × Ratio`
  - 例如: GPU 显存分配了 80GB 用于 KV Cache，设置 `--hicache-ratio 2.0`，则 CPU 内存中申请 **160GB** 作为 L2 Cache

### 2. **Hybrid KV Cache Manager（混合注意力支持）**

根据 SGLang 文档，SGLang 也支持 **Hybrid KV Cache Manager**，专门为混合注意力机制模型设计（如 Llama 4，结合了 local chunked attention 和 full attention 层）。

**功能：**
- 为不同 attention 类型的层分配不同的 cache slots
- 支持层特定的 prefix-cache 规则
- 优化混合模型的内存使用

**对 Llama-4-Scout 的影响：**
- **8xH100**: 启用后可以支持 **5M tokens**（从 1M 提升）
- **8xH200**: 启用后可以支持 **10M tokens**（从 2.5M 提升）

## 与 vLLM 的 Hybrid KV Cache Manager 对比

| 特性 | vLLM Hybrid KV Cache Manager | SGLang Hybrid KV Cache |
|------|------------------------------|------------------------|
| **主要目的** | 优化混合注意力模型的内存使用（按层分配） | 扩展 KV cache 容量（GPU/CPU 分层） |
| **工作原理** | Sliding window 层只保留窗口内 tokens，Full attention 层保留全部 | GPU/CPU 内存分层存储 |
| **启用方式** | 环境变量 `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1` | `--enable-hierarchical-cache` + `--hicache-ratio <float>` |
| **对 Llama-4-Scout** | 从 2.94M 提升到 11.6M tokens（8xH200） | 从 2.5M 提升到 10M tokens（8xH200） |
| **性能影响** | 可能有延迟回归（latency regression） | 需要 CPU-GPU 数据传输，可能有延迟 |

## 当前配置状态

### vLLM（已启用 Hybrid KV Cache Manager）
- ✅ 已通过环境变量启用：`VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1`
- ✅ 测试成功：4.91M tokens（5M 配置）
- ✅ 理论最大：11.6M tokens per request（2.96x concurrency）
- ✅ 8M 配置：1.86x concurrency for 8M tokens

### SGLang（未启用 HiCache）
- ❌ 当前配置**未启用** `--enable-hierarchical-cache` 和 `--hicache-ratio`
- ⚠️ 当前只测试了 2M context length
- 📝 根据文档，启用后可以支持 10M tokens（8xH200）

## 建议：测试 SGLang 的 Hybrid KV Cache

### 1. 启用 SGLang 的 HiCache

修改 `run-sglang-docker.sh`，添加 `--enable-hierarchical-cache` 和 `--hicache-ratio` 参数：

```bash
python3 -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 8 \
  --context-length 5242880 \  # 5M tokens
  --mem-fraction-static 0.80 \
  --disable-cuda-graph \
  --enable-hierarchical-cache \  # 启用 HiCache（必须）
  --hicache-ratio 2.0 \  # CPU 内存是 GPU 显存的 2 倍（默认值）
  --trust-remote-code
```

**参数说明：**
- `--enable-hierarchical-cache`: **必须项**，启用分层缓存
- `--hicache-ratio 2.0`: CPU 内存相对于 GPU 显存的比例（默认 2.0，可根据系统内存调整）
- `--hicache-write-policy write_through`: (可选) 写入策略，默认为 `write_through`（直写模式，有助于多轮对话的 Cache 命中率）

### 2. 测试更大的 Context Length

根据文档，启用后可以测试：
- **5M tokens**（8xH100 的推荐值）
- **10M tokens**（8xH200 的理论最大值）

### 3. 性能对比

测试启用 Hybrid KV Cache 后的：
- **最大支持的 context length**
- **Prompt throughput**
- **延迟影响**（CPU-GPU 数据传输）

## 关键区别总结

1. **vLLM 的 Hybrid KV Cache Manager**：
   - 专注于**按层优化内存**（sliding window vs full attention）
   - 所有 KV cache 仍在 GPU 上
   - 通过减少 sliding window 层的 KV cache 占用来提升容量

2. **SGLang 的 HiCache（Hierarchical Cache）**：
   - 专注于**扩展容量**（GPU + CPU 分层）
   - 部分 KV cache 存储在 CPU 内存（L2 Cache）
   - 通过 CPU 内存扩展来支持更大的 context length
   - 需要启用 `--enable-hierarchical-cache` 和设置 `--hicache-ratio`

3. **两者可以结合使用**：
   - vLLM: 启用 Hybrid Manager（按层优化）+ 增加 `gpu-memory-utilization`
   - SGLang: 启用 `--enable-hierarchical-cache` + `--hicache-ratio`（GPU/CPU 分层）

## 参考资料

- [SGLang Llama4 Documentation](https://docs.sglang.io/basic_usage/llama4.html)
- [SGLang HiCache Documentation](https://docs.sglang.ai/advanced_features/hicache.html)
- [vLLM Hybrid KV Cache Manager Documentation](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)
