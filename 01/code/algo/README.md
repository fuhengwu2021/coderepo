# AllReduce Communication Algorithms

This directory contains implementations of various AllReduce communication algorithms.

## Ring AllReduce

Ring AllReduce is a bandwidth-optimal algorithm that works in two phases:

### Phase 1: Scatter-Reduce
- Data is split into chunks
- Chunks circulate the ring topology
- Each rank reduces chunks as they pass through
- After (world_size - 1) steps, each rank has one fully reduced chunk

### Phase 2: AllGather
- Reduced chunks circulate the ring again
- Each rank receives and forwards chunks
- After (world_size - 1) steps, all ranks have all reduced chunks

### Complexity
- Communication steps: 2 × (world_size - 1)
- Bandwidth: Optimal (each element is sent exactly twice)
- Latency: O(world_size)

### Usage
```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_ring_allreduce.py
```

## Other Algorithms (To be implemented)

- Tree AllReduce
- Double Binary Tree AllReduce
- Recursive Doubling AllReduce

