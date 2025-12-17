#!/bin/bash
#SBATCH --job-name=train-distributed
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --output=train_%j_%N.out
#SBATCH --error=train_%j_%N.err

# This script runs distributed training with train.py across 2 nodes (node6 and node7)
# Each node gets 1 GPU, for a total of 2 GPUs
#
# Usage:
#   sbatch train_sbatch.sh
#
# Logs:
#   - Each node writes to separate log files:
#     - train_<job_id>_node6.out/err (for node6, rank 0)
#     - train_<job_id>_node7.out/err (for node7, rank 1)
#   - View logs: tail -f train_<job_id>_node6.out
#   - View all logs: tail -f train_<job_id>_*.out
#   - srun output is prefixed with [rank<N>]: to identify which rank produced each line

echo "=========================================="
echo "Distributed Training with train.py"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Current node: $SLURMD_NODENAME"
echo ""

# Get master node address - use localhost for single-node multi-GPU setup
# For true multi-node, you would use the first node's hostname
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Set distributed training environment variables
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo ""

# Display node and GPU information
echo "Node information:"
srun hostname
echo ""

echo "GPU information:"
srun nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# NCCL settings
export NCCL_DEBUG=WARN  # Set to INFO for more details
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

echo "=========================================="
echo "Starting training on node $SLURMD_NODENAME (rank $RANK)..."
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run distributed training using srun
# Each rank will write to its own log file (train_<job_id>_<node_name>.out/err)
# The --label option prefixes each line with [rank<N>]: to identify the source rank
srun --label python train.py \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --dataset-size 1000 \
    --input-dim 10

echo ""
echo "=========================================="
echo "Training completed on node $SLURMD_NODENAME (rank $RANK)!"
echo "=========================================="
