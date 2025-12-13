# mdaisy

Distributed computing utilities and demos for PyTorch.

## Installation

```bash
cd shared
pip install -e .
```

## Usage

### With torchrun (Recommended)

```bash
torchrun --nproc_per_node=4 your_script.py
```

```python
from mdaisy import init_distributed, sync_print

# Initialize distributed training (reads from torchrun environment)
rank, world_size, device, local_rank = init_distributed(use_cpu=False)

# Use synchronized printing to avoid interleaved output
sync_print(f"Rank {rank} says hello!", rank=rank, world_size=world_size)
```

### Without torchrun (Alternative)

```python
from mdaisy import run_distributed

def my_worker(rank, world_size, device, local_rank):
    # Your distributed code here
    print(f"Rank {rank} running on {device}")

# Run with 4 processes
run_distributed(my_worker, world_size=4, use_cpu=False)
```

See `example_no_torchrun.py` for a complete example.

## Features

- **init_distributed()**: Initialize distributed process groups with automatic GPU/CPU fallback
- **sync_print()**: Synchronized printing that prevents output interleaving across ranks
- **run_distributed()**: Launch distributed code without torchrun using multiprocessing

## Development

Install in development mode:

```bash
pip install -e .[dev]
```

