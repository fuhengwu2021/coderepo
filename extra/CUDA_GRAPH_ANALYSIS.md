# CUDA Graph 利弊分析

## CUDA Graph 的好处

### 1. **性能提升**
- **减少 CPU 开销**：将多个 GPU kernel 操作捕获为一个图，减少 kernel launch 的开销
- **提高吞吐量**：减少 kernel 之间的间隙，提高整体执行效率
- **更一致的延迟**：GPU 内部处理依赖关系，减少 CPU-GPU 交互带来的延迟波动

### 2. **适用场景**
- **静态输入形状**：当输入形状固定时效果最好
- **重复执行模式**：相同的操作序列多次执行
- **小 batch size**：对于小 batch，kernel launch 开销占比更大，收益更明显

## CUDA Graph 的代价

### 1. **内存开销（最重要）**
- **预分配缓冲区**：需要为所有可能的输入形状预分配内存
- **对于 2M context**：可能需要额外 4-10GB 内存用于 graph 缓冲区
- **内存碎片**：可能导致内存碎片化

### 2. **启动时间**
- **Graph 捕获**：需要先执行一次来捕获操作序列（warmup）
- **编译时间**：graph 的编译和优化需要时间
- **对于 2M context**：捕获过程可能需要几分钟

### 3. **灵活性限制**
- **固定形状**：每个 graph 只能处理特定的输入形状
- **动态输入**：如果输入形状变化，需要重新捕获 graph
- **大 context**：对于超大 context（如 2M），可能需要多个 graph 变体

## 对于 2M Context 的建议

### 禁用 CUDA Graph 的原因：

1. **内存限制**
   - 2M context 已经需要 ~384GB KV cache
   - CUDA graph 额外需要 4-10GB 内存（每个 GPU）
   - 总共可能需要 ~50GB+ 每 GPU，接近 H200 的 143GB 限制

2. **启动时间**
   - Graph 捕获对于 2M context 可能需要很长时间
   - 禁用后启动更快

3. **性能权衡**
   - 对于 2M context，kernel launch 开销相对较小（因为每个 kernel 处理的数据量大）
   - 性能损失可能只有 5-15%，但可以节省大量内存

### 建议配置：

```bash
# SGLang with CUDA graph disabled (for 2M context)
--disable-cuda-graph
--mem-fraction-static 0.80  # 保守的内存使用
```

### 性能影响估算：

- **启用 CUDA graph**：可能提升 10-20% 吞吐量，但需要额外 4-10GB 内存
- **禁用 CUDA graph**：性能可能降低 5-15%，但节省大量内存，更稳定

### 结论：

对于 **2M context length**，**建议禁用 CUDA graph**：
- ✅ 节省内存（避免 OOM）
- ✅ 更快的启动时间
- ✅ 更稳定（避免内存碎片）
- ⚠️ 轻微的性能损失（5-15%），但对于大 context 可接受
