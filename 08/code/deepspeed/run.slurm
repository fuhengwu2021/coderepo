#!/bin/bash
#SBATCH --job-name=ds-zero3-offload
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=logs/ds_zero3_%j_%N.out
#SBATCH --error=logs/ds_zero3_%j_%N.err

# This script runs DeepSpeed ZeRO-3 with CPU offload training
# Multi-Node Single-GPU Setup: Multiple nodes, each with 1 GPU
#
# Usage:
#   mkdir -p logs
#   sbatch run.slurm
#
# Logs:
#   - Logs are written to the logs/ directory
#   - Each node writes to separate log files

# Activate conda environment
# Initialize conda (adjust path if needed)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate research

# Get the directory where this script is located
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "=========================================="
echo "DeepSpeed ZeRO-3 with CPU Offload Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Current node: $SLURMD_NODENAME"
echo "Logs directory: $SCRIPT_DIR/logs"
echo ""

# --------- Required distributed environment variables ---------
# Get master node address from SLURM (first node in the list)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# NCCL settings
export NCCL_DEBUG=WARN  # Set to INFO for more details
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

# OpenMP settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment variables:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  SLURM_PROCID: $SLURM_PROCID"
echo "  SLURM_LOCALID: $SLURM_LOCALID"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo ""

# Display node and GPU information
echo "Node information:"
srun hostname
echo ""

echo "GPU information:"
srun nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
echo ""

# Check DeepSpeed installation
echo "Checking DeepSpeed installation:"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')" || {
    echo "ERROR: DeepSpeed not installed. Install with: pip install deepspeed"
    exit 1
}
echo ""

# Check if config file exists
CONFIG_FILE="$SCRIPT_DIR/ds_zero3_offload.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: DeepSpeed config file not found: $CONFIG_FILE"
    exit 1
fi
echo "Using DeepSpeed config: $CONFIG_FILE"
echo ""

# Check if training script exists
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi
echo "Training script: $TRAIN_SCRIPT"
echo ""

echo "=========================================="
echo "Starting DeepSpeed training..."
echo "=========================================="
echo ""

# --------- Correct way to launch DeepSpeed with SLURM ---------
# Use srun with deepspeed command
# DeepSpeed automatically reads SLURM_PROCID and other SLURM env vars
# Do NOT use torchrun - DeepSpeed handles distributed setup internally
# --num_gpus should be per node (1 in this case), not total
srun --chdir="$SCRIPT_DIR" --label \
    deepspeed \
    --num_gpus=$SLURM_NTASKS_PER_NODE \
    --num_nodes=$SLURM_JOB_NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "$TRAIN_SCRIPT" \
    --deepspeed \
    --deepspeed-config "$CONFIG_FILE" \
    --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
    --steps 10

echo ""
echo "=========================================="
echo "DeepSpeed training completed!"
echo "=========================================="
