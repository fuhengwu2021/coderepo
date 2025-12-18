# Llama-4-Scout Deployment and Testing

Deployment and testing configurations for **Llama-4-Scout-17B-16E-Instruct** with vLLM and SGLang on **8x H200 GPUs** with **2M context length** (2,097,152 tokens).

## Overview

This directory contains scripts and configurations to test if vLLM and SGLang can handle Llama-4-Scout with 2M context length on 8x H200 GPUs, as required for production deployment.

**Test Requirements:**
- Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Context size: 2M tokens (2,097,152)
- Output length: 200 tokens
- Hardware: 8x H200 GPUs
- Backends: vLLM v0.12.0 and SGLang v0.5.6.post2-runtime

## Test Results

### ✅ vLLM v0.12.0 - SUCCESS

**Configuration:**
- Image: `vllm/vllm-openai:v0.12.0`
- Tensor Parallel Size: 8
- Max Model Length: 2,097,152 tokens
- GPU Memory Utilization: 0.9

**Test Results:**
- ✅ Successfully processed **2.07M tokens input** + 200 tokens output
- **Prompt throughput**: **206,527.9 tokens/s** (excellent performance for 2M context!)
- **Generation throughput**: **20.0 tokens/s**
- **Prefix cache hit rate**: **30.2%** (cache optimization working, improves performance)
- **Response time**: **69.35 seconds** for 2.07M tokens + 200 output
- **Status**: **200 OK** ✅

**Performance Analysis:**
- Processing 2M+ tokens in ~70 seconds demonstrates vLLM can handle large contexts efficiently
- 206K tokens/s prompt throughput is excellent for such large context lengths
- Prefix cache (30.2% hit rate) helps optimize repeated content processing

**Token Generation Strategy:**
- Uses **smart sampling**: tokenizer samples first 100K characters to estimate actual ratio (~4.07 chars/token)
- Uses sampled ratio directly (no buffer) to avoid exceeding 2M limit
- Actual result: **2,065,427 tokens** (slightly over 2M by ~3%, server accepts with small tolerance)
- The server supports 2M context length as configured (`--max-model-len 2097152`)
- **Smart sampling is optimal**: 
  - Fast: only samples 100K chars (takes ~1-2 seconds)
  - Accurate: uses actual tokenizer ratio
  - Safe: avoids significantly exceeding 2M limit

**Conclusion:** vLLM v0.12.0 **works** for Llama-4-Scout with 2M context length on 8x H200.

## Model Path

The model is located at:
```
/mnt/co-research/shared-models/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct
```

**Note:** Use HuggingFace model ID `meta-llama/Llama-4-Scout-17B-16E-Instruct` in configurations. vLLM will automatically resolve it from the HF_HOME cache.

## KV Cache Memory Requirements

For **2M context length** with Llama-4-Scout-17B-16E-Instruct:

| Item | Size |
|------|------|
| **Per token KV cache** | 0.1875 MB |
| **Total KV cache (all GPUs)** | 384 GB |
| **Per GPU KV cache (8-way TP)** | 48 GB |
| **Model weights (per GPU)** | ~4 GB |
| **Total per GPU** | ~52 GB |

**Key Optimization:** The model uses **GQA (Grouped Query Attention)** with 8 KV heads instead of 40, reducing KV cache by **80%** (from 1920 GB to 384 GB).

## Files

### Docker Deployment (Local - Recommended)
- `run-vllm-docker.sh` - Run vLLM with Docker
- `run-sglang-docker.sh` - Run SGLang with Docker

### Kubernetes Deployment
- `vllm-llama-4-scout.yaml` - Kubernetes config for vLLM
- `sglang-llama-4-scout.yaml` - Kubernetes config for SGLang
- `deploy-vllm-llama-4-scout.sh` - Kubernetes deployment script for vLLM
- `deploy-sglang-llama-4-scout.sh` - Kubernetes deployment script for SGLang

### Testing
- `test_llama4_scout.py` - Test script for 2M context + 200 output tokens
- `load_llama4_scout.py` - Direct model loading script
- `run-test.sh` - Wrapper script (activates conda env "research")
- `run-load.sh` - Wrapper script for load script

### Data
- `large_text_10mb.txt` - Large text file (15.7MB) for generating 2M token inputs

## Prerequisites

1. **Conda Environment (Required):**
   ```bash
   conda create -n research python=3.10
   conda activate research
   pip install requests transformers
   ```
   
   **Important:** All test scripts automatically activate the `research` conda environment. You don't need to manually activate it when using the wrapper scripts (`run-test.sh`, `run-load.sh`).

2. **HF_TOKEN (if required):**
   ```bash
   export HF_TOKEN='your_token_here'
   ```

3. **Docker with GPU support:**
   - 8x H200 GPUs accessible via `--gpus all`
   - Model path mounted at `/mnt/co-research/shared-models`

## Quick Start

### 1. Deploy vLLM Server

```bash
cd /home/fuhwu/workspace/coderepo/extra
./run-vllm-docker.sh
```

**Expected startup time:** 8-10 minutes
- Model loading: ~8 minutes
- KV cache initialization: ~2 minutes
- Total: ~10 minutes

**Monitor logs:**
```bash
docker logs -f vllm-llama-4-scout
```

Wait for: `Application startup complete.`

### 2. Test with 2M Context

```bash
# Using wrapper script (activates conda env automatically)
./run-test.sh --backend vllm --input-length 2097152 --output-length 200
```

**Expected results:**
- ✅ Request succeeds (200 OK)
- Prompt throughput: ~160K tokens/s
- Generation throughput: ~10-20 tokens/s
- KV cache usage: ~40-50%

### 3. Test SGLang (Optional)

```bash
# Deploy SGLang
./run-sglang-docker.sh

# Test
./run-test.sh --backend sglang --input-length 2097152 --output-length 200
```

## Configuration Details

### vLLM Configuration
- **Image**: `vllm/vllm-openai:v0.12.0`
- **Tensor Parallel Size**: 8 (8x H200)
- **Max Model Length**: 2,097,152 tokens (2M)
- **GPU Memory Utilization**: 0.9
- **Entrypoint**: `python3 -m vllm.entrypoints.openai.api_server`

### SGLang Configuration
- **Image**: `lmsysorg/sglang:v0.5.6.post2-runtime`
- **Tensor Parallel Size**: 8 (8x H200)
- **Context Length**: 2,097,152 tokens (2M)

## Testing Different Context Lengths

```bash
# Test with 10K tokens
./run-test.sh --backend vllm --input-length 10000 --output-length 200

# Test with 100K tokens
./run-test.sh --backend vllm --input-length 100000 --output-length 200

# Test with 2M tokens (full test)
./run-test.sh --backend vllm --input-length 2097152 --output-length 200
```

## Monitoring

### Check Container Status
```bash
docker ps | grep llama-4-scout
```

### View Logs
```bash
# vLLM
docker logs -f vllm-llama-4-scout

# SGLang
docker logs -f sglang-llama-4-scout
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Check Service Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

## Troubleshooting

### Container Exits Immediately
- Check logs: `docker logs vllm-llama-4-scout`
- Verify GPU access: `nvidia-smi`
- Check model path exists

### Model Path Error
- Use HuggingFace model ID: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- Ensure `HF_HOME` is set correctly
- Verify model is in cache at `/mnt/co-research/shared-models/hub`

### Out of Memory
- Reduce `--gpu-memory-utilization` (default: 0.9)
- Check if other processes are using GPU memory
- Verify all 8 GPUs are available

### Slow Startup
- **Normal:** 8-10 minutes for 2M context is expected
- KV cache allocation for 2M tokens takes time
- Monitor logs for progress

## Performance Notes

### Why 2M Context Takes Time
1. **KV Cache Allocation**: ~384 GB total KV cache needs initialization
2. **Model Loading**: 17B parameters across 8 GPUs
3. **Compilation**: torch.compile optimization on first run
4. **MoE Architecture**: 16 experts add complexity

### Expected Performance
- **Prompt Processing**: ~160K tokens/s
- **Generation**: ~10-20 tokens/s (depends on output length)
- **Memory Usage**: ~133GB per GPU (out of 143GB available)

## Key Findings

1. ✅ **vLLM v0.12.0 works** with Llama-4-Scout at 2M context
2. ✅ **GQA optimization** reduces KV cache by 80%
3. ✅ **PagedAttention** enables efficient memory management
4. ✅ **8x H200** provides sufficient memory (133GB used / 143GB total)
5. ✅ **Test passed**: 1.62M tokens input + 200 tokens output

## Next Steps

1. **Test SGLang** with same configuration
2. **Concurrency testing**: 50 concurrent requests (as per requirements)
3. **Variable context testing**: 10K to 2M tokens
4. **Production deployment**: Use Kubernetes configs if needed

## References

- Model: [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- vLLM: [vllm.ai](https://vllm.ai)
- SGLang: [sglang.ai](https://sglang.ai)
