"""
TP Model Wrapper with Real Weight Loading (vLLM-style)

This module provides a wrapper that loads real model weights and applies TP,
demonstrating how vLLM loads and shards weights for inference.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict
import sys
import os

try:
    from .parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from .linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
    from .mlp import TensorParallelMLP
    from .weight_loader import load_and_shard_state_dict, apply_tp_to_model_weights
except ImportError:
    from parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
    from mlp import TensorParallelMLP
    from weight_loader import load_and_shard_state_dict, apply_tp_to_model_weights


class TPModelWrapper:
    """
    Wrapper that loads real model weights and applies TP (vLLM-style).
    
    This demonstrates the key difference from generic TP:
    - Loads weights from HuggingFace checkpoint
    - Shards weights during loading (not after)
    - Only stores sharded weights in memory
    - Works with real model architectures
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize TP model wrapper with real weight loading.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            dtype: Data type for weights (default: from model config)
        """
        self.model_name = model_name
        self.device = device
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_main_rank = (self.tp_rank == 0)
        
        if self.is_main_rank:
            print(f"Loading model {model_name} with TP={self.tp_size}...")
        
        # Load config
        self.config = AutoConfig.from_pretrained(model_name)
        if dtype is None:
            # Try to get dtype from config (handle both torch_dtype and dtype)
            dtype = getattr(self.config, 'torch_dtype', None)
            if dtype is None:
                dtype = getattr(self.config, 'dtype', torch.float32)
        self.dtype = dtype
        
        # Get model dimensions
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_layers = self.config.num_hidden_layers
        self.intermediate_size = getattr(self.config, 'intermediate_size', 4 * self.hidden_size)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load full state dict (on main rank, or all ranks if needed)
        # In production vLLM, only main rank loads, then broadcasts metadata
        # For simplicity, all ranks load here
        if self.is_main_rank:
            print("Loading weights from checkpoint...")
        
        self.state_dict = load_and_shard_state_dict(
            model_name,
            device="cpu",  # Load on CPU first, then move to device
            dtype=dtype,
        )
        
        if self.is_main_rank:
            print(f"Loaded {len(self.state_dict)} weight tensors")
            print(f"Applying TP sharding (each rank will store 1/{self.tp_size} of weights)...")
    
    def create_tp_attention_layer(self, layer_idx: int) -> nn.Module:
        """
        Create a TP-aware attention layer and load weights.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            TP attention layer with loaded weights
        """
        try:
            from .attention import TensorParallelAttention
        except ImportError:
            from attention import TensorParallelAttention
        
        attn = TensorParallelAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bias=False,
            params_dtype=self.dtype,
        ).to(self.device)
        
        # Load weights from state dict
        prefix = f"model.layers.{layer_idx}.self_attn"
        
        # Try separate Q, K, V first (Qwen uses separate projections)
        q_key = f"{prefix}.q_proj.weight"
        k_key = f"{prefix}.k_proj.weight"
        v_key = f"{prefix}.v_proj.weight"
        
        if all(k in self.state_dict for k in [q_key, k_key, v_key]):
            # Load separately (Qwen-style)
            attn.qkv_proj.weight_loader_qkv(self.state_dict[q_key], "q")
            attn.qkv_proj.weight_loader_qkv(self.state_dict[k_key], "k")
            attn.qkv_proj.weight_loader_qkv(self.state_dict[v_key], "v")
        else:
            # Try fused QKV
            qkv_key = f"{prefix}.qkv_proj.weight"
            if qkv_key in self.state_dict:
                # Fused QKV
                attn.qkv_proj.weight_loader_qkv(self.state_dict[qkv_key], loaded_shard_id=None)
        
        # Load output projection
        o_key = f"{prefix}.o_proj.weight"
        if o_key in self.state_dict:
            attn.o_proj.weight_loader(self.state_dict[o_key])
        
        return attn
    
    def create_tp_mlp_layer(self, layer_idx: int) -> TensorParallelMLP:
        """
        Create a TP-aware MLP layer and load weights.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            TP MLP layer with loaded weights
        """
        mlp = TensorParallelMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=self.dtype,
        ).to(self.device)
        
        # Load weights from state dict
        prefix = f"model.layers.{layer_idx}.mlp"
        
        # Gate projection
        gate_key = f"{prefix}.gate_proj.weight"
        if gate_key in self.state_dict:
            mlp.gate_proj.weight_loader(self.state_dict[gate_key])
        
        # Up projection
        up_key = f"{prefix}.up_proj.weight"
        if up_key in self.state_dict:
            mlp.up_proj.weight_loader(self.state_dict[up_key])
        
        # Down projection
        down_key = f"{prefix}.down_proj.weight"
        if down_key in self.state_dict:
            mlp.down_proj.weight_loader(self.state_dict[down_key])
        
        return mlp
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics (showing TP memory savings).
        
        Returns:
            Dictionary with memory stats
        """
        total_params = 0
        total_memory_mb = 0
        
        # Count parameters in TP layers
        # This is simplified - in reality, we'd iterate through the model
        # For demo purposes, calculate expected memory
        
        # Attention layers: QKV + O
        attn_params_per_layer = (
            self.hidden_size * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim +  # QKV
            self.hidden_size * self.hidden_size  # O
        ) // self.tp_size
        
        # MLP layers: Gate + Up + Down
        mlp_params_per_layer = (
            self.hidden_size * self.intermediate_size +  # Gate
            self.hidden_size * self.intermediate_size +  # Up
            self.intermediate_size * self.hidden_size  # Down
        ) // self.tp_size
        
        total_params = (attn_params_per_layer + mlp_params_per_layer) * self.num_layers
        total_memory_mb = (total_params * 2) / (1024 * 1024)  # 2 bytes per float16
        
        return {
            "total_parameters": total_params,
            "memory_mb": total_memory_mb,
            "tp_size": self.tp_size,
            "memory_per_rank_mb": total_memory_mb,
        }
