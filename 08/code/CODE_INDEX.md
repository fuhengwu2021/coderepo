# Chapter 8 Code Index

This directory contains all code examples from Chapter 8, configured for **Multi-Node Single-GPU Setup** (one GPU per node).

## Python Training Scripts

### 1. `train.py`
- **Description**: Main distributed training script with automatic single/multi-GPU detection
- **Features**: 
  - Automatically detects single vs multi-GPU mode
  - Works with SLURM environment variables
  - Supports both DDP and single-GPU training
- **Usage**: 
  ```bash
  srun -N 1 --gres=gpu:1 python train.py                    # Single GPU (single node)
  srun -N 2 --gres=gpu:1 python train.py                   # Multi-GPU (2 nodes, 1 GPU each)
  ```

### 2. `train_ddp.py`
- **Description**: PyTorch DDP (Distributed Data Parallel) training script
- **Features**: 
  - Full DDP implementation
  - Automatic SLURM environment detection
  - Works with multi-node single-GPU setup (one GPU per node)
- **Usage**: See `train_ddp.sh` or run directly with torchrun

### 3. `train_fsdp.py`
- **Description**: PyTorch FSDP (Fully Sharded Data Parallel) training script
- **Features**: 
  - FSDP implementation for large models
  - CPU offload support
  - Automatic wrapping policy
- **Usage**: See `train_fsdp.sh` or run directly with torchrun

### 4. `train_distributed.py`
- **Description**: Complete distributed training script with checkpointing
- **Features**: 
  - Full training workflow
  - Checkpoint saving and resumption
  - Error handling
  - Command-line arguments
- **Usage**: See `train_distributed.sh`

### 5. `checkpoint.py`
- **Description**: Checkpoint utility script
- **Features**: 
  - Manual checkpoint saving
  - Useful for handling preemption signals
- **Usage**: 
  ```bash
  python checkpoint.py --checkpoint_dir ./checkpoints
  ```

## Framework-Specific Examples

### DeepSpeed (`deepspeed/`)
- **Description**: DeepSpeed ZeRO-3 with CPU offload training example
- **Files**: 
  - `train.py` - Training script with DeepSpeed integration
  - `run.slurm` - SLURM batch script
  - `ds_zero3_offload.json` - DeepSpeed configuration
  - `README.md` - Detailed documentation
- **Features**:
  - ZeRO-3 parameter sharding
  - CPU offload for large models
  - Automatic distributed setup
  - Works with HuggingFace models
- **Usage**: 
  ```bash
  cd deepspeed
  mkdir -p logs
  sbatch run.slurm
  ```

### Megatron-LM (`megatron/`)
- **Description**: Megatron-LM GPT pretraining with SLURM
- **Files**:
  - `run.slurm` - SLURM batch script for Megatron training
  - `README.md` - Detailed documentation
- **Features**:
  - Tensor, pipeline, and data parallelism support
  - Multi-node training
  - Mock data support for testing
  - Configurable model size
- **Prerequisites**:
  - Megatron-LM repository installed
  - Set `MEGATRON_PATH` environment variable
- **Usage**:
  ```bash
  cd megatron
  export MEGATRON_PATH=/path/to/megatron-lm
  mkdir -p logs
  sbatch run.slurm
  ```

## Shell Scripts (SLURM Batch Jobs)

### 1. `train_ddp.sh`
- **Description**: SLURM batch script for DDP training
- **Configuration**: 
  - 2 nodes (`--nodes=2`)
  - 1 GPU per node (`--gres=gpu:1`)
  - 1 task per node (`--ntasks-per-node=1`)
- **Usage**: 
  ```bash
  sbatch train_ddp.sh
  ```

### 2. `train_fsdp.sh`
- **Description**: SLURM batch script for FSDP training
- **Configuration**: 
  - 2 nodes (`--nodes=2`)
  - 1 GPU per node (`--gres=gpu:1`)
  - 1 task per node (`--ntasks-per-node=1`)
- **Usage**: 
  ```bash
  sbatch train_fsdp.sh
  ```

### 3. `train_array.sh`
- **Description**: SLURM job array script for hyperparameter tuning
- **Features**: 
  - Runs 10 jobs (array 0-9)
  - Different learning rates and batch sizes per job
- **Usage**: 
  ```bash
  sbatch train_array.sh
  ```

### 4. `train_distributed.sh`
- **Description**: Complete distributed training batch script with checkpointing
- **Features**: 
  - Signal handling for preemption
  - Checkpoint resumption
  - Full configuration
- **Usage**: 
  ```bash
  sbatch train_distributed.sh
  ```

## Multi-Node Single-GPU Configuration

All scripts are configured for **multi-node single-GPU** setup (one GPU per node):

- **Nodes**: `--nodes=2` (multiple nodes)
- **GPUs**: `--gres=gpu:1` (one GPU per node)
- **Tasks**: `--ntasks-per-node=1` (one task per node)

### Example: Running with 4 GPUs

To use 4 GPUs, you need 4 nodes (one GPU per node):

```bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
```

Then in the torchrun command:
```bash
torchrun --nproc_per_node=1 --nnodes=4 ...
```

## Quick Start

### 1. Single GPU Training
```bash
srun -N 1 --gres=gpu:1 python train.py
```

### 2. Multi-GPU Training (2 nodes, 1 GPU each)
```bash
sbatch train_ddp.sh
```

### 3. Hyperparameter Tuning
```bash
sbatch train_array.sh
```

### 4. Full Training with Checkpointing
```bash
sbatch train_distributed.sh
```

## File Mapping to Chapter 8

| Chapter Section | Code File |
|----------------|-----------|
| 3.1 Basic Job Submission | `train.py` |
| 3.2 Batch Jobs | `train_ddp.sh` |
| 4.1 PyTorch DDP | `train_ddp.py`, `train_ddp.sh` |
| 4.2 PyTorch FSDP | `train_fsdp.py`, `train_fsdp.sh` |
| 4.3 DeepSpeed ZeRO-3 | `deepspeed/train.py`, `deepspeed/run.slurm` |
| 4.4 Megatron-LM | `megatron/run.slurm` |
| 5.1 Job Arrays | `train_array.sh` |
| 5.4 Checkpointing | `checkpoint.py`, `train_distributed.sh` |
| 9. Complete Workflow | `train_distributed.py`, `train_distributed.sh` |

## Notes

- All scripts automatically detect SLURM environment variables
- Scripts work with both `srun` (interactive) and `sbatch` (batch) modes
- Single-GPU mode is automatically handled (no distributed setup needed)
- All scripts include proper cleanup and error handling
- NCCL settings are configured for common network issues

## Testing

To test the setup:

```bash
# Check SLURM setup
sinfo

# Test single GPU
srun -N 1 --gres=gpu:1 python train.py --epochs 2

# Test multi-GPU (2 nodes, 1 GPU each)
srun -N 2 --gres=gpu:1 --ntasks-per-node=1 torchrun --nproc_per_node=1 --nnodes=2 train_ddp.py
```
