#!/bin/bash
# Test script to verify CPU mode works
# This will run the demo with CPU backend

echo "Testing Tensor Parallelism with CPU mode..."
echo "This requires at least 2 CPU processes"
echo ""

# Run with 2 processes using CPU (gloo backend)
torchrun --nproc_per_node=2 demo.py --force-cpu

