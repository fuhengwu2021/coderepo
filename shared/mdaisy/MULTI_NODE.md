# Multi-Node Distributed Training with torchrun

Yes, `torchrun` supports multi-node distributed training! Here's how to use it.

## Basic Multi-Node Setup

### Single Node (what we've been using)
```bash
torchrun --nproc_per_node=4 demo_reduce.py
```

### Multi-Node Setup

```bash
# On node 0 (master node)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    demo_reduce.py

# On node 1 (worker node)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    demo_reduce.py
```

## Parameters Explained

- `--nnodes=N`: Total number of nodes (e.g., 2 for 2 nodes)
- `--nproc_per_node=M`: Number of processes per node (e.g., 4 GPUs per node)
- `--node_rank=R`: Rank of this node (0 for master, 1, 2, ... for workers)
- `--master_addr="IP"`: IP address of the master node (node 0)
- `--master_port=P`: Port for communication (default: 29500)

## Environment Variables in Multi-Node

When using multi-node, `torchrun` sets:

- **RANK**: Global rank across all nodes (0, 1, 2, ..., 7 for 2 nodes × 4 GPUs)
- **WORLD_SIZE**: Total processes across all nodes (8 for 2 nodes × 4 GPUs)
- **LOCAL_RANK**: Rank within the node (0, 1, 2, 3 on each node)
- **LOCAL_WORLD_SIZE**: Processes per node (4 in this example)

### Example with 2 nodes, 4 GPUs each:

**Node 0:**
- Process 0: RANK=0, LOCAL_RANK=0
- Process 1: RANK=1, LOCAL_RANK=1
- Process 2: RANK=2, LOCAL_RANK=2
- Process 3: RANK=3, LOCAL_RANK=3

**Node 1:**
- Process 0: RANK=4, LOCAL_RANK=0
- Process 1: RANK=5, LOCAL_RANK=1
- Process 2: RANK=6, LOCAL_RANK=2
- Process 3: RANK=7, LOCAL_RANK=3

## Using with SLURM

If you're using SLURM, you can use `srun` to launch on multiple nodes:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Launch on all nodes
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    demo_reduce.py
```

## Using with SSH (Manual Setup)

If you have SSH access to multiple nodes:

**On master node (node 0):**
```bash
# Get master node IP
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# Launch on this node
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    demo_reduce.py
```

**On worker node (node 1):**
```bash
# Use the same MASTER_ADDR from node 0
MASTER_ADDR="192.168.1.100"  # IP of node 0

torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=29500 \
    demo_reduce.py
```

## Important Notes

1. **Firewall**: Ensure the master port (default 29500) is open between nodes
2. **Network**: Nodes must be able to communicate via TCP/IP
3. **NCCL Backend**: For GPU training, NCCL backend is recommended for multi-node
4. **Master Node**: The master node (node_rank=0) coordinates initialization

## Testing Multi-Node

You can test multi-node setup even on a single machine by using different ports:

```bash
# Terminal 1 (simulating node 0)
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
    --master_addr="localhost" --master_port=29500 \
    demo_reduce.py

# Terminal 2 (simulating node 1)  
torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 \
    --master_addr="localhost" --master_port=29500 \
    demo_reduce.py
```

This will run 4 processes total (2 per "node") on a single machine.

