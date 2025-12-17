"""
Tensor Parallel MLP Layer (vLLM-style)
Demonstrates how to apply TP to a Multi-Layer Perceptron (MLP)

An MLP layer consists of:
Input → Up Projection (Column Parallel) → Activation → Down Projection (Row Parallel) → Output
"""
import torch
import torch.nn as nn
from typing import Optional

# Try relative import first (when used as package), fallback to absolute (when used as script)
try:
    from .linear import ColumnParallelLinear, RowParallelLinear
except ImportError:
    from linear import ColumnParallelLinear, RowParallelLinear


class SiLU(nn.Module):
    """SiLU activation function: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class TensorParallelMLP(nn.Module):
    """
    Tensor Parallel MLP Layer (vLLM-style)
    
    This demonstrates the key insight from the chapter:
    - Up Projection uses Column Parallelism (splits along output dimension)
    - Activation operates on sharded data (no communication needed)
    - Down Projection uses Row Parallelism (splits along input dimension)
    - We avoid all-gather after column parallel because row parallel needs sharded data
    - Only all-reduce is needed at the end
    
    Args:
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size (typically 4x hidden_size for LLaMA)
        params_dtype: Data type for parameters (default: float32)
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, params_dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        if params_dtype is None:
            params_dtype = torch.float32
        
        # Up Projection: Column Parallel
        # Input: [..., hidden_size]
        # Weight: [hidden_size, intermediate_size] -> sharded along columns
        # Output: [..., intermediate_size_per_partition] (sharded)
        self.gate_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            gather_output=False,  # Don't gather, next layer needs sharded data
            params_dtype=params_dtype,
        )
        
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=False,
            gather_output=False,  # Don't gather, next layer needs sharded data
            params_dtype=params_dtype,
        )
        
        # Activation: Element-wise operation on sharded data
        self.activation = SiLU()
        
        # Down Projection: Row Parallel
        # Input: [..., intermediate_size_per_partition] (sharded)
        # Weight: [intermediate_size, hidden_size] -> sharded along rows
        # Output: [..., hidden_size] (after all-reduce)
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            input_is_parallel=True,  # Input is already sharded
            reduce_results=True,  # All-reduce to get final result
            params_dtype=params_dtype,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through tensor parallel MLP
        
        Args:
            x: Input tensor of shape [..., hidden_size]
        
        Returns:
            Output tensor of shape [..., hidden_size]
        """
        # Step 1: Up Projection (Column Parallel)
        # Each GPU computes: x @ gate_proj_shard and x @ up_proj_shard
        # Results are sharded: [..., intermediate_size_per_partition]
        gate = self.gate_proj(x)  # Sharded output
        up = self.up_proj(x)  # Sharded output
        
        # Step 2: Activation (element-wise, operates on sharded data)
        # No communication needed
        activated = self.activation(gate) * up  # Sharded output
        
        # Step 3: Down Projection (Row Parallel)
        # Each GPU computes: activated_shard @ down_proj_shard
        # All-reduce sums the partial results
        # Final output: [..., hidden_size] (same on all GPUs)
        output = self.down_proj(activated)
        
        return output
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.gate_proj.input_size}, intermediate_size={self.gate_proj.output_size}"


def create_mlp(hidden_size: int, intermediate_size: int) -> TensorParallelMLP:
    """Factory function to create a tensor parallel MLP"""
    return TensorParallelMLP(hidden_size, intermediate_size)

