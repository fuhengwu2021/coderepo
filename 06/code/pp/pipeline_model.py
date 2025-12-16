"""
Pipeline Parallel Model Implementation
Splits a model across multiple pipeline stages
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PipelineStage(nn.Module):
    """A single stage in a pipeline parallel model"""
    
    def __init__(self, layers: List[nn.Module], stage_idx: int, num_stages: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.stage_idx = stage_idx
        self.num_stages = num_stages
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through this stage's layers"""
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """A transformer block that can be split across pipeline stages"""
    
    def __init__(self, hidden_size: int = 128, num_heads: int = 4, intermediate_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # MLP
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        
        batch_size, seq_len, hidden_size = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output
        
        # MLP
        residual = x
        x = self.mlp_norm(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_output = self.down_proj(gate * up)
        x = residual + mlp_output
        
        return x


def create_pipeline_stages(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    num_stages: int
) -> List[List[nn.Module]]:
    """
    Create pipeline stages by splitting layers across stages.
    
    Args:
        num_layers: Total number of transformer layers
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        num_stages: Number of pipeline stages
        
    Returns:
        List of lists, where each inner list contains modules for one stage
    """
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    
    stages = []
    layer_idx = 0
    
    for stage_idx in range(num_stages):
        # Distribute remainder layers to earlier stages
        num_layers_this_stage = layers_per_stage + (1 if stage_idx < remainder else 0)
        
        stage_layers = []
        for _ in range(num_layers_this_stage):
            stage_layers.append(
                TransformerBlock(hidden_size, num_heads, intermediate_size)
            )
        
        stages.append(stage_layers)
        layer_idx += num_layers_this_stage
    
    return stages


class PipelineParallelModel:
    """
    A model split across pipeline stages.
    Each rank holds one stage and communicates activations.
    """
    
    def __init__(
        self,
        stage: PipelineStage,
        stage_idx: int,
        num_stages: int,
        hidden_size: int,
        device: torch.device
    ):
        self.stage = stage.to(device)
        self.stage_idx = stage_idx
        self.num_stages = num_stages
        self.hidden_size = hidden_size
        self.device = device
        
        # Import here to avoid circular dependency
        from parallel_state import (
            get_prev_rank, get_next_rank, 
            is_first_stage, is_last_stage
        )
        self.prev_rank = get_prev_rank()
        self.next_rank = get_next_rank()
        self.is_first = is_first_stage()
        self.is_last = is_last_stage()
    
    def forward(
        self, 
        x: Optional[torch.Tensor] = None,
        recv_from_prev: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Forward pass through this pipeline stage.
        
        Args:
            x: Input tensor (only used for first stage)
            recv_from_prev: Whether to receive input from previous stage
            
        Returns:
            Output tensor (None for non-last stages if not collecting output)
        """
        import torch.distributed as dist
        from parallel_state import get_pp_group
        
        pp_group = get_pp_group()
        
        # Receive input from previous stage (if not first stage)
        if not self.is_first and recv_from_prev:
            if self.prev_rank is not None:
                # Receive shape and dtype info first (simplified - in practice use metadata)
                # For simplicity, we assume fixed shapes
                if x is None:
                    # In a real implementation, we'd receive metadata first
                    # For demo, we'll use a placeholder
                    raise ValueError("Non-first stage must receive input from previous stage")
        
        # Forward through this stage
        output = self.stage(x)
        
        # Send output to next stage (if not last stage)
        if not self.is_last and self.next_rank is not None:
            pp_group.send(output, dst=self.next_rank)
            # Return None for intermediate stages (output sent to next stage)
            return None
        
        # Last stage returns the output
        return output

