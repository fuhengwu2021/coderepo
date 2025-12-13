"""
mdaisy - Distributed computing utilities and demos for PyTorch.

This package provides utilities for distributed PyTorch operations including:
- Distributed initialization helpers
- Synchronized printing utilities
- Collective operation demos
- Point-to-point communication demos
"""

from .utils import init_distributed, sync_print, run_distributed, get_node_info

__version__ = "0.1.0"
__all__ = ['init_distributed', 'sync_print', 'run_distributed', 'get_node_info']

