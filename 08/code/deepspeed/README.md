# DeepSpeed ZeRO-3 with CPU Offload

This directory contains a complete DeepSpeed ZeRO-3 with CPU offload implementation for Single-Node Multi-GPU setup.

## Overview

This implementation demonstrates DeepSpeed's unique capability to train models larger than GPU memory by:
- **ZeRO-3**: Sharding parameters, gradients, and optimizer states across GPUs
- **CPU Offload**: Offloading parameters and optimizer states to CPU memory
- **Automatic Distributed Setup**: DeepSpeed handles all distributed training setup automatically

## Files

1. **`train.py`** - Training script using HuggingFace models with DeepSpeed
2. **`ds_zero3_offload.json`** - DeepSpeed configuration for ZeRO-3 with CPU offload
3. **`run.slurm`** - SLURM batch script for single-node multi-GPU setup

## Key Features

### Why DeepSpeed ZeRO-3 + CPU Offload?

- **DDP cannot do this**: DDP requires full model replica on each GPU
- **FSDP2 doesn't support CPU offload**: FSDP2 focuses on GPU memory optimization
- **Megatron requires code changes**: Megatron needs model architecture modifications
- **DeepSpeed is built for this**: Runtime parameter sharding and CPU offload without code changes

### What This Enables

- Train models larger than total GPU memory
- Use standard HuggingFace models without modification
- Automatic distributed setup (no manual rank/world_size management)
- Efficient memory usage with CPU offload

## Configuration

### Single-Node Multi-GPU Setup

The `run.slurm` script is configured for:
- **1 node** (`--nodes=1`)
- **2 GPUs** (`--gres=gpu:2`) - adjust as needed
- **2 tasks** (`--ntasks-per-node=2`) - one per GPU

To use more GPUs, modify:
```bash
#SBATCH --gres=gpu:4        # 4 GPUs
#SBATCH --ntasks-per-node=4 # 4 tasks
```

### DeepSpeed Configuration

The `ds_zero3_offload.json` includes:
- **ZeRO Stage 3**: Full sharding of parameters, gradients, and optimizer states
- **Parameter Offload**: Offloads model parameters to CPU
- **Optimizer Offload**: Offloads optimizer states to CPU
- **FP16**: Mixed precision training
- **Communication Overlap**: Overlaps communication with computation

## Usage

### 1. Install Dependencies

```bash
pip install deepspeed transformers torch
```

### 2. Prepare Logs Directory

```bash
cd code/deepspeed
mkdir -p logs
```

### 3. Submit Job

```bash
sbatch run.slurm
```

### 4. Monitor Training

```bash
# Watch job queue
squeue -u $USER

# View logs
tail -f logs/ds_zero3_<job_id>_<node>.out
```

## Verification

When training starts, you should see in the logs:

```
============================================================
DeepSpeed ZeRO-3 Initialization Complete
============================================================
  ZeRO Stage: 3
  Parameter offloading: True
  Optimizer offloading: True
  Global rank: 0
  World size: 2
  Device: cuda:0
============================================================
```

### Expected Behavior

1. **GPU Memory**: Should be significantly lower than model size
2. **CPU Memory**: Will increase (parameters and optimizer states offloaded)
3. **Training**: Should proceed normally with loss decreasing
4. **No OOM Errors**: Even with models larger than GPU memory

## Customization

### Change Model

Edit `run.slurm` or pass as argument:
```bash
--model-name "meta-llama/Llama-2-7b-hf"
```

### Adjust Training Steps

Edit `run.slurm`:
```bash
--steps 100
```

### Modify DeepSpeed Config

Edit `ds_zero3_offload.json`:
- Change `train_batch_size` for larger batches
- Adjust `gradient_accumulation_steps` for effective batch size
- Modify offload settings if needed

## Troubleshooting

### DeepSpeed Not Found

```bash
pip install deepspeed
# Or with CUDA support:
DS_BUILD_OPS=1 pip install deepspeed
```

### CUDA Out of Memory

- Reduce `train_micro_batch_size_per_gpu` in config
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Verify CPU offload is enabled (check logs)

### Communication Errors

- Check `NCCL_SOCKET_IFNAME` is set correctly
- Verify `MASTER_ADDR` and `MASTER_PORT` are accessible
- Check firewall settings

## Comparison with Other Methods

| Feature | DDP | FSDP | FSDP2 | DeepSpeed ZeRO-3 |
|---------|-----|------|-------|------------------|
| Parameter Sharding | ❌ | ✅ | ✅ | ✅ |
| Gradient Sharding | ❌ | ✅ | ✅ | ✅ |
| Optimizer Sharding | ❌ | ✅ | ✅ | ✅ |
| CPU Offload | ❌ | ✅ | ❌ | ✅ |
| No Code Changes | ✅ | ✅ | ✅ | ✅ |
| HuggingFace Support | ✅ | ✅ | ✅ | ✅ |

## References

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
