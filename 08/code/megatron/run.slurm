#!/bin/bash
#SBATCH --job-name=megatron-gpt
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_%j_%N.out
#SBATCH --error=logs/train_%j_%N.err

# This script runs Megatron-LM GPT pretraining with SLURM
# Multi-Node Single-GPU Setup: Multiple nodes, each with 1 GPU
#
# Usage:
#   1. Install megatron-core: pip install --no-build-isolation megatron-core[mlm,dev]
#   2. Ensure pretrain_gpt.py, gpt_builders.py, and model_provider.py are in this directory
#   3. mkdir -p logs
#   4. sbatch run.slurm
#
# Logs:
#   - Logs are written to the logs/ directory
#   - Each node writes to separate log files

# Activate conda environment
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate research || {
    echo "ERROR: Failed to activate conda environment 'research'"
    exit 1
}

# Get the directory where this script is located
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
cd "$SCRIPT_DIR"
mkdir -p logs

# Use pretrain_gpt.py from the same directory as this script
# The script and required files (gpt_builders.py, model_provider.py) are in this directory
PRETRAIN_SCRIPT="${SCRIPT_DIR}/pretrain_gpt.py"

if [ ! -f "$PRETRAIN_SCRIPT" ]; then
    echo "ERROR: pretrain_gpt.py not found at $PRETRAIN_SCRIPT"
    echo "Please ensure pretrain_gpt.py is in the same directory as this script"
    exit 1
fi

# Check for required helper files
if [ ! -f "${SCRIPT_DIR}/gpt_builders.py" ] || [ ! -f "${SCRIPT_DIR}/model_provider.py" ]; then
    echo "ERROR: Required files (gpt_builders.py, model_provider.py) not found in $SCRIPT_DIR"
    echo "Please ensure all required files are present"
    exit 1
fi

echo "=========================================="
echo "Megatron-LM GPT Pretraining with SLURM"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Current node: $SLURMD_NODENAME"
echo "Script directory: $SCRIPT_DIR"
echo "Logs directory: $SCRIPT_DIR/logs"
echo ""

# --------- Distributed training setup ---------
# Get master node address from SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-6000}
export WORLD_SIZE=$SLURM_NTASKS

# NCCL settings
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export CUDA_DEVICE_MAX_CONNECTIONS=1

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

# --------- Training configuration ---------
# Checkpoint and log paths
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/gpt_8b}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-${SCRIPT_DIR}/tensorboard_logs/gpt_8b}"

# Create directories
mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"

# Data cache path
DATA_CACHE_PATH="${DATA_CACHE_PATH:-${SCRIPT_DIR}/data_cache}"
mkdir -p "$DATA_CACHE_PATH"

# Model configuration (smaller model for demonstration)
NUM_LAYERS=${NUM_LAYERS:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-14336}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
SEQ_LENGTH=${SEQ_LENGTH:-2048}  # Reduced for smaller memory footprint
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-2048}

# Parallelism configuration
TP_SIZE=${TP_SIZE:-1}      # Tensor parallelism
CP_SIZE=${CP_SIZE:-1}       # Context parallelism
PP_SIZE=${PP_SIZE:-1}       # Pipeline parallelism

# Training hyperparameters
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LR=${LR:-0.00015}
MIN_LR=${MIN_LR:-0.00001}

# Data configuration (use mock data for demonstration)
USE_MOCK_DATA=${USE_MOCK_DATA:-1}  # Set to 0 to use real data
TOKENIZER_ARG=${TOKENIZER_ARG:-"MOCK"}
DATA_ARG=${DATA_ARG:-"MOCK"}

echo "Training configuration:"
echo "  Checkpoint path: $CHECKPOINT_PATH"
echo "  TensorBoard logs: $TENSORBOARD_LOGS_PATH"
echo "  Data cache: $DATA_CACHE_PATH"
echo "  Model: ${NUM_LAYERS} layers, hidden_size=${HIDDEN_SIZE}"
echo "  Parallelism: TP=${TP_SIZE}, CP=${CP_SIZE}, PP=${PP_SIZE}"
echo "  Batch size: micro=${MICRO_BATCH_SIZE}, global=${GLOBAL_BATCH_SIZE}"
echo "  Learning rate: $LR (min: $MIN_LR)"
echo "  Use mock data: $USE_MOCK_DATA"
echo ""

# --------- Build training command ---------
# Model arguments
MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

# Training arguments
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 1000000  # Reduced for demonstration
    --lr-decay-samples 950000
    --lr-warmup-samples 5000
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --bf16
    --grad-reduce-in-bf16
    --cross-entropy-loss-fusion
    --calculate-per-token-loss
    --manual-gc
    --empty-unused-memory-level 1
)

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --sequence-parallel
)

# Distributed Data Parallel arguments
DDP_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
TRAINING_ARGS+=("${DDP_ARGS[@]}")

# Data arguments
DATA_ARGS_LIST=()
if [[ "$USE_MOCK_DATA" == "1" ]] || [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]]; then
    DATA_ARGS_LIST+=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 128256
        --data-cache-path "${DATA_CACHE_PATH}"
        --tiktoken-pattern v2
        --split '99,1,0'
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
    )
else
    DATA_ARGS_LIST+=(
        --data-path "$DATA_ARG"
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_ARG"
        --data-cache-path "${DATA_CACHE_PATH}"
        --split '99,1,0'
        --no-create-attention-mask-in-dataloader
        --no-mmap-bin-files
        --num-workers 1
        --vocab-size 128256
    )
fi

# Evaluation and logging arguments
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-iters 10
    --eval-interval 100
    --save-interval 500
    --log-throughput
    --ckpt-format torch_dist
    --distributed-timeout-minutes 60
    --save "$CHECKPOINT_PATH"
    --load "$CHECKPOINT_PATH"
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
)

echo "=========================================="
echo "Starting Megatron training..."
echo "=========================================="
echo ""

# Build the complete command with all arguments
# Convert arrays to space-separated strings for passing to bash -c
MODEL_ARGS_STR="${MODEL_ARGS[*]}"
TRAINING_ARGS_STR="${TRAINING_ARGS[*]}"
MODEL_PARALLEL_ARGS_STR="${MODEL_PARALLEL_ARGS[*]}"
DATA_ARGS_STR="${DATA_ARGS_LIST[*]}"
EVAL_ARGS_STR="${EVAL_AND_LOGGING_ARGS[*]}"

# Use srun to launch training on each node
# Each srun task runs one Python process via torchrun
srun --chdir="$SCRIPT_DIR" --label \
    bash -c "
        NODE=\$(hostname)
        echo \"[Node \$NODE] Activating conda environment...\"
        source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
            source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
        conda activate research || {
            echo \"[Node \$NODE] ERROR: Failed to activate conda environment\"
            exit 1
        }
        
        # Set CUDA_VISIBLE_DEVICES for this node's GPU
        if [[ \"\$SLURMD_NODENAME\" =~ ^node([0-9]+)\$ ]]; then
            GPU_NUM=\${BASH_REMATCH[1]}
        else
            GPU_NUM=\$SLURM_LOCALID
        fi
        export CUDA_VISIBLE_DEVICES=\$GPU_NUM
        export LOCAL_RANK=\$SLURM_LOCALID
        export RANK=\$SLURM_PROCID
        
        echo \"[Node \$NODE] CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES\"
        echo \"[Node \$NODE] RANK: \$RANK, LOCAL_RANK: \$LOCAL_RANK\"
        echo \"[Node \$NODE] Starting training...\"
        
        # Change to script directory (where pretrain_gpt.py is located)
        cd \"$SCRIPT_DIR\"
        
        # Launch training with torchrun
        # For multi-node single-GPU setup, each node runs one process
        # torchrun handles distributed initialization
        torchrun \
            --nproc_per_node=1 \
            --nnodes=\$SLURM_JOB_NUM_NODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=\"$MASTER_ADDR\" \
            --master_port=\"$MASTER_PORT\" \
            \"$PRETRAIN_SCRIPT\" \
            $MODEL_ARGS_STR \
            $TRAINING_ARGS_STR \
            $MODEL_PARALLEL_ARGS_STR \
            $DATA_ARGS_STR \
            $EVAL_ARGS_STR
    "

echo ""
echo "=========================================="
echo "Megatron training completed!"
echo "=========================================="
