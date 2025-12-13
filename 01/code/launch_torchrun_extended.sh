#!/usr/bin/env bash
# Launch extended multi-GPU example with CIFAR-10 and more epochs

# Set OMP_NUM_THREADS to control OpenMP threads per process
# Must be set BEFORE torchrun to avoid the warning message
export OMP_NUM_THREADS=4

# Run with 2 GPUs, 20 epochs by default
torchrun --nproc_per_node=2 code/multi_gpu_ddp_extended.py --epochs 20
