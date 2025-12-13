# Installation Instructions

## Install mdaisy package

From the `shared/` directory:

```bash
cd shared
pip install -e .
```

This installs the package in editable mode, so changes to the code will be reflected immediately.

## Usage in demo files

All demo files now import from mdaisy:

```python
import sys
import os

# Add shared directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../shared'))
from mdaisy import init_distributed, sync_print

# Use the functions
rank, world_size, device, local_rank = init_distributed(use_cpu=False)
sync_print(f"Rank {rank} says hello!", rank=rank, world_size=world_size)
```

## Package Structure

```
shared/
├── mdaisy/
│   ├── __init__.py      # Package initialization, exports main functions
│   └── utils.py         # Shared utilities (init_distributed, sync_print)
├── setup.py             # Package setup configuration
├── README.md            # Package documentation
└── .gitignore          # Git ignore rules
```

## Functions Available

### `init_distributed(use_cpu=False)`
Initialize distributed process group with automatic GPU/CPU fallback.

**Returns:** `(rank, world_size, device, local_rank)`

### `sync_print(*args, rank=None, world_size=1, **kwargs)`
Synchronized printing that prevents output interleaving across ranks.

**Parameters:**
- `rank`: Current process rank (required if world_size > 1)
- `world_size`: Total number of processes

