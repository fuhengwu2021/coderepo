"""
Weight Loading Utilities for vLLM-style TP

This module provides utilities to load and shard model weights from HuggingFace
checkpoints for tensor parallelism, matching vLLM's weight loading approach.
"""
import torch
from typing import Dict, Optional
from transformers import AutoModel

try:
    from .parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
except ImportError:
    from parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size


def load_and_shard_state_dict(
    model_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load full model state dict from HuggingFace checkpoint.
    
    Args:
        model_path: Path to model (HuggingFace model name or local path)
        device: Device to load weights on
        dtype: Data type for weights (default: from model config)
    
    Returns:
        Full state dict with all weights
    """
    rank = get_tensor_model_parallel_rank()
    if rank == 0:
        print(f"Loading model weights from {model_path}...")
    
    # Load model config first to get dtype if not specified
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    if dtype is None:
        # Try to infer from config
        if hasattr(config, 'torch_dtype'):
            dtype = config.torch_dtype
        elif hasattr(config, 'dtype'):
            dtype = config.dtype
        else:
            dtype = torch.float32
    
    # Load state dict using safetensors if available, otherwise use torch
    try:
        from safetensors.torch import load_file
        import os
        
        # Try to find safetensors files
        model_files = []
        if os.path.isdir(model_path):
            for f in os.listdir(model_path):
                if f.endswith(".safetensors"):
                    model_files.append(os.path.join(model_path, f))
        
        if model_files:
            # Load from safetensors
            state_dict = {}
            for f in model_files:
                state_dict.update(load_file(f))
            if rank == 0:
                print(f"Loaded {len(state_dict)} tensors from safetensors")
        else:
            # Fallback to torch.load
            model = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
                device_map="cpu",  # Load on CPU first
            )
            state_dict = model.state_dict()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if rank == 0:
                print(f"Loaded {len(state_dict)} tensors from PyTorch checkpoint")
    except ImportError:
        # No safetensors, use torch.load
        model = AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cpu",
        )
        state_dict = model.state_dict()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if rank == 0:
            print(f"Loaded {len(state_dict)} tensors from PyTorch checkpoint")
    
    # Move to specified device
    if device != "cpu":
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
    return state_dict


def get_weight_name_mapping(model_name: str) -> Dict[str, str]:
    """
    Get mapping from standard layer names to HuggingFace checkpoint names.
    
    This handles different naming conventions (e.g., LLaMA vs Qwen).
    
    Args:
        model_name: Model name to determine naming convention
    
    Returns:
        Dictionary mapping standard names to checkpoint names
    """
    # Common patterns
    if "qwen" in model_name.lower():
        return {
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "o_proj": "o_proj",
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        }
    elif "llama" in model_name.lower():
        return {
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "o_proj": "o_proj",
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        }
    else:
        # Default: assume standard naming
        return {
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "o_proj": "o_proj",
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        }


def load_weight_for_layer(
    layer,
    state_dict: Dict[str, torch.Tensor],
    weight_key: str,
    bias_key: Optional[str] = None,
):
    """
    Load and shard weight for a TP layer from state dict.
    
    Args:
        layer: TP layer (ColumnParallelLinear, RowParallelLinear, etc.)
        state_dict: Full state dict from checkpoint
        weight_key: Key for weight in state dict
        bias_key: Optional key for bias in state dict
    """
    if weight_key not in state_dict:
        raise KeyError(f"Weight key '{weight_key}' not found in state dict")
    
    loaded_weight = state_dict[weight_key]
    
    # Use layer's weight_loader method
    if hasattr(layer, 'weight_loader'):
        layer.weight_loader(loaded_weight)
    elif hasattr(layer, 'weight_loader_qkv'):
        # For QKV layers, try to detect if fused or separate
        layer.weight_loader_qkv(loaded_weight, loaded_shard_id=None)
    else:
        # Fallback: direct assignment (shouldn't happen with proper layers)
        raise AttributeError(f"Layer {type(layer)} has no weight_loader method")
    
    # Load bias if present
    if bias_key and bias_key in state_dict and hasattr(layer, 'bias_loader'):
        loaded_bias = state_dict[bias_key]
        layer.bias_loader(loaded_bias)
    elif bias_key and bias_key in state_dict and layer.bias is not None:
        # Fallback: direct assignment
        layer.bias.data.copy_(state_dict[bias_key])


def apply_tp_to_model_weights(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    model_name: str = "",
):
    """
    Apply TP weight loading to a model that uses TP layers.
    
    This function iterates through the model and loads weights into TP layers
    using their weight_loader methods.
    
    Args:
        model: Model with TP layers
        state_dict: Full state dict from checkpoint
        model_name: Model name for naming convention detection
    """
    name_mapping = get_weight_name_mapping(model_name)
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight_loader') or hasattr(module, 'weight_loader_qkv'):
            # Try to find corresponding weight in state dict
            # Handle different naming patterns
            for pattern, checkpoint_name in name_mapping.items():
                if pattern in name.lower():
                    # Construct full key (e.g., "model.layers.0.self_attn.q_proj.weight")
                    weight_key = f"{name}.weight"
                    bias_key = f"{name}.bias"
                    
                    if weight_key in state_dict:
                        load_weight_for_layer(module, state_dict, weight_key, bias_key)
                        break
