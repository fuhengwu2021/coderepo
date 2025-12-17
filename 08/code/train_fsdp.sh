#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=train_fsdp_%j.out
#SBATCH --error=train_fsdp_%j.err

# Get node list and master address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker,lo

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE, Rank: $RANK, Local rank: $LOCAL_RANK"
echo "Node list: $SLURM_JOB_NODELIST"

# Run FSDP training with torchrun
torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_fsdp.py
