#!/usr/bin/env python3
"""
GPU Memory Calculator for Deep Learning Models

This script calculates GPU memory requirements for training and inference
based on model architecture, precision, optimizer, and training configuration.

Example usage:
    python gpu_memory_calculator.py --model meta-llama/Meta-Llama-3-8B --mode training --optimizer adamw
"""

import argparse
import gc
import sys
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")

# Constants
GB_TO_BYTES = 1024 ** 3
DEFAULT_VOCAB_SIZE = 32000
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_NUM_LAYERS = 32
DEFAULT_NUM_HEADS = 32
DEFAULT_SEQ_LENGTH = 2048
DEFAULT_FFN_MULTIPLIER = 4  # FFN intermediate_size = hidden_size * multiplier
GRADIENT_CHECKPOINTING_MULTIPLIER = 0.3  # Reduces activation memory by ~70%
INFERENCE_ACTIVATION_RATIO = 0.1  # Inference activations are ~10% of training
ATTENTION_PROJECTIONS = 4  # Q, K, V, O
KV_CACHE_PROJECTIONS = 2  # K and V
A100_80GB = 80.0
A100_40GB = 40.0
H100_80GB = 80.0


class Precision(Enum):
    """Model precision types"""
    FP32 = (4, "FP32")  # bytes per parameter
    FP16 = (2, "FP16")
    BF16 = (2, "BF16")
    INT8 = (1, "INT8")
    INT4 = (0.5, "INT4")
    
    def __init__(self, bytes_per_param, display_name):
        self.bytes_per_param = bytes_per_param
        self.display_name = display_name
    
    @property
    def value(self):
        """Return bytes per parameter"""
        return self.bytes_per_param


class OptimizerType(Enum):
    """Optimizer types and their state memory requirements"""
    SGD = (0.0, "SGD")
    SGD_MOMENTUM = (1.0, "SGD with Momentum")
    NESTEROV = (1.0, "Nesterov")
    ADAGRAD = (1.0, "Adagrad")
    RMSPROP = (1.0, "RMSProp")
    ADAM = (2.0, "Adam")
    ADAMW = (2.0, "AdamW")
    ADAFACTOR = (0.5, "Adafactor")
    LAMB = (2.0, "LAMB")
    LION = (1.0, "Lion")
    NADAM = (2.0, "NAdam")
    AMSGRAD = (2.0, "AMSGrad")
    SPARSE_ADAM = (2.0, "SparseAdam")
    SHAMPOO = (3.0, "Shampoo")
    ADABELIEF = (2.0, "AdaBelief")
    
    def __init__(self, multiplier, display_name):
        self.multiplier = multiplier
        self.display_name = display_name
    
    @property
    def value(self):
        """Return the memory multiplier"""
        return self.multiplier


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    num_params: int  # Total number of parameters
    vocab_size: int = DEFAULT_VOCAB_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    num_heads: int = DEFAULT_NUM_HEADS
    seq_length: int = DEFAULT_SEQ_LENGTH
    intermediate_size: Optional[int] = None  # FFN intermediate size
    
    def __post_init__(self):
        """Set default intermediate size if not provided"""
        if self.intermediate_size is None:
            self.intermediate_size = DEFAULT_FFN_MULTIPLIER * self.hidden_size


def load_model_config_from_transformers(model_name: str, load_full_model: bool = False) -> ModelConfig:
    """
    Load model configuration from transformers library.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'meta-llama/Meta-Llama-3-8B')
    
    Returns:
        ModelConfig object with model parameters
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    print(f"Loading model configuration from HuggingFace: {model_name}")
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Extract model parameters
        vocab_size = getattr(config, 'vocab_size', DEFAULT_VOCAB_SIZE)
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', DEFAULT_HIDDEN_SIZE))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', getattr(config, 'num_layers', DEFAULT_NUM_LAYERS)))
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', getattr(config, 'num_heads', DEFAULT_NUM_HEADS)))
        
        # Get intermediate size (FFN dimension)
        intermediate_size = getattr(config, 'intermediate_size', getattr(config, 'ffn_dim', getattr(config, 'd_ff', None)))
        
        # Get max position embeddings (for default seq_length)
        max_seq_length = getattr(config, 'max_position_embeddings', getattr(config, 'max_seq_len', DEFAULT_SEQ_LENGTH))
        
        # Calculate number of parameters using meta device
        # Meta device creates tensors with shape/dtype metadata only, no actual memory allocation
        if load_full_model:
            print("Loading full model to count parameters (this may take a moment and use significant memory)...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype='auto',
                    device_map='cpu',
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                num_params = sum(p.numel() for p in model.parameters())
                del model
                gc.collect()
                print(f"Loaded model and counted {num_params:,} parameters")
            except Exception as e:
                print(f"Error: Could not load full model to count parameters: {e}")
                print("Exiting.")
                sys.exit(1)
        else:
            # Use meta device to create model structure without allocating memory
            print("Counting parameters using meta device (zero memory allocation)...")
            try:
                import torch
                # Check if meta device is available
                try:
                    test_tensor = torch.empty(1, device='meta')
                    del test_tensor
                except Exception:
                    print("Error: Meta device is not supported in this PyTorch version.")
                    print("Please upgrade to PyTorch 2.0+ or use --load-full-model for accurate parameter count.")
                    sys.exit(1)
                
                # Create model on meta device (no actual memory allocation)
                with torch.device('meta'):
                    model = AutoModelForCausalLM.from_config(
                        config,
                        trust_remote_code=True
                    )
                    num_params = sum(p.numel() for p in model.parameters())
                    del model
                
                gc.collect()
                print(f"Counted {num_params:,} parameters using meta device")
            except Exception as e:
                print(f"Error: Failed to count parameters using meta device: {e}")
                print("This may happen if:")
                print("  1. The model uses operations not supported on meta device")
                print("  2. Custom layers don't support meta device")
                print("  3. PyTorch version doesn't fully support meta device")
                print("\nTry using --load-full-model for accurate parameter count (requires more memory).")
                sys.exit(1)
        
        # Create ModelConfig
        model_config = ModelConfig(
            name=model_name.split('/')[-1] if '/' in model_name else model_name,
            num_params=num_params,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            seq_length=max_seq_length,
            intermediate_size=intermediate_size,
        )
        
        print(f"Loaded configuration:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Intermediate size: {model_config.intermediate_size}")
        print(f"  Max sequence length: {max_seq_length}")
        
        return model_config
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "403" in error_msg or "restricted" in error_msg.lower():
            raise ValueError(
                f"Model '{model_name}' is a gated/restricted model and requires authorization.\n"
                f"To access it:\n"
                f"1. Visit https://huggingface.co/{model_name} and request access\n"
                f"2. Login to HuggingFace: huggingface-cli login\n"
                f"3. Accept the model's terms of use on the model page\n"
                f"4. Try again after authorization\n\n"
                f"Alternatively, use a publicly available model like:\n"
                f"  - microsoft/phi-2\n"
                f"  - meta-llama/Llama-2-7b-hf (if you have access)\n"
                f"  - mistralai/Mistral-7B-v0.1\n"
            )
        else:
            raise ValueError(
                f"Failed to load model configuration for '{model_name}': {e}\n"
                f"Make sure the model name is correct and accessible from HuggingFace Hub.\n"
                f"If the model is gated, you may need to login: huggingface-cli login"
            )


# Legacy model name mappings for convenience
LEGACY_MODEL_NAMES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
}


class GPUMemoryCalculator:
    """Calculate GPU memory requirements for model training/inference"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        precision: Precision = Precision.BF16,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        optimizer: OptimizerType = OptimizerType.ADAMW,
        use_gradient_checkpointing: bool = False,
        kv_cache_enabled: bool = True,  # For inference
    ):
        self.model_config = model_config
        self.precision = precision
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.kv_cache_enabled = kv_cache_enabled
        
        # Bytes per parameter based on precision
        self.bytes_per_param = precision.value
    
    def calculate_model_weights(self) -> float:
        """Calculate memory for model weights in GB"""
        return (self.model_config.num_params * self.bytes_per_param) / GB_TO_BYTES
    
    def calculate_gradients(self) -> float:
        """Calculate memory for gradients in GB"""
        # Gradients are same size as weights
        return (self.model_config.num_params * self.bytes_per_param) / GB_TO_BYTES
    
    def calculate_optimizer_states(self) -> float:
        """Calculate memory for optimizer states in GB"""
        optimizer_multiplier = self.optimizer.value
        return (self.model_config.num_params * self.bytes_per_param * optimizer_multiplier) / GB_TO_BYTES
    
    def calculate_activation_memory(self) -> float:
        """
        Calculate activation memory for training.
        
        For transformer models, activation memory includes:
        - Attention activations: batch_size * seq_length * hidden_size * num_layers
        - FFN activations: batch_size * seq_length * intermediate_size * num_layers
        - Layer norm activations
        - Residual connections
        
        Simplified estimation: ~2 * batch_size * seq_length * hidden_size * num_layers
        """
        activation_multiplier = GRADIENT_CHECKPOINTING_MULTIPLIER if self.use_gradient_checkpointing else 1.0
        
        # Rough estimation: activations scale with batch, sequence, and model size
        # Each layer stores activations of size batch_size * seq_length * hidden_size
        # Plus FFN intermediate activations
        base_activation = (
            self.batch_size * 
            self.model_config.seq_length * 
            self.model_config.hidden_size * 
            self.model_config.num_layers
        )
        
        # Add FFN intermediate activations
        ffn_activation = (
            self.batch_size *
            self.model_config.seq_length *
            self.model_config.intermediate_size *
            self.model_config.num_layers
        )
        
        # Attention activations (Q, K, V, output)
        attention_activation = (
            self.batch_size *
            self.model_config.seq_length *
            self.model_config.hidden_size *
            self.model_config.num_layers *
            ATTENTION_PROJECTIONS
        )
        
        total_activation_elements = (base_activation + ffn_activation + attention_activation) * activation_multiplier
        
        # Activations are typically stored in the same precision as model
        return (total_activation_elements * self.bytes_per_param) / GB_TO_BYTES
    
    def calculate_kv_cache(self) -> float:
        """
        Calculate KV cache memory for inference.
        
        KV cache stores key-value pairs for attention mechanism.
        For each layer: batch_size * seq_length * hidden_size * 2 (K and V)
        """
        if not self.kv_cache_enabled:
            return 0.0
        
        kv_cache_per_layer = (
            self.batch_size *
            self.model_config.seq_length *
            self.model_config.hidden_size *
            KV_CACHE_PROJECTIONS
        )
        
        total_kv_cache = kv_cache_per_layer * self.model_config.num_layers
        
        return (total_kv_cache * self.bytes_per_param) / GB_TO_BYTES
    
    def calculate_training_memory(self) -> Dict[str, float]:
        """Calculate total memory for training"""
        weights = self.calculate_model_weights()
        gradients = self.calculate_gradients()
        optimizer_states = self.calculate_optimizer_states()
        activations = self.calculate_activation_memory()
        
        # Input embeddings (vocab_size * hidden_size)
        embedding_memory = (
            self.model_config.vocab_size *
            self.model_config.hidden_size *
            self.bytes_per_param
        ) / GB_TO_BYTES
        
        # Input data (batch_size * seq_length * hidden_size)
        input_memory = (
            self.batch_size *
            self.model_config.seq_length *
            self.model_config.hidden_size *
            self.bytes_per_param
        ) / GB_TO_BYTES
        
        total = weights + gradients + optimizer_states + activations + embedding_memory + input_memory
        
        return {
            "Model Weights": weights,
            "Gradients": gradients,
            "Optimizer States": optimizer_states,
            "Activations": activations,
            "Embeddings": embedding_memory,
            "Input Data": input_memory,
            "Total (GB)": total,
        }
    
    def calculate_inference_memory(self) -> Dict[str, float]:
        """Calculate total memory for inference"""
        weights = self.calculate_model_weights()
        kv_cache = self.calculate_kv_cache()
        
        # Input embeddings
        embedding_memory = (
            self.model_config.vocab_size *
            self.model_config.hidden_size *
            self.bytes_per_param
        ) / GB_TO_BYTES
        
        # Input data
        input_memory = (
            self.batch_size *
            self.model_config.seq_length *
            self.model_config.hidden_size *
            self.bytes_per_param
        ) / GB_TO_BYTES
        
        # Temporary activations (much smaller than training)
        temp_activations = self.calculate_activation_memory() * INFERENCE_ACTIVATION_RATIO
        
        total = weights + kv_cache + embedding_memory + input_memory + temp_activations
        
        return {
            "Model Weights": weights,
            "KV Cache": kv_cache,
            "Embeddings": embedding_memory,
            "Input Data": input_memory,
            "Temp Activations": temp_activations,
            "Total (GB)": total,
        }
    
    def print_memory_breakdown(self, mode: str = "training"):
        """Print detailed memory breakdown"""
        if mode == "training":
            memory = self.calculate_training_memory()
        else:
            memory = self.calculate_inference_memory()
        
        print(f"\n{'='*70}")
        print(f"GPU Memory Breakdown: {self.model_config.name}")
        print(f"{'='*70}")
        print(f"Mode: {mode.upper()}")
        print(f"Precision: {self.precision.name}")
        print(f"Batch Size: {self.batch_size}")
        if mode == "training":
            print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            print(f"Optimizer: {self.optimizer.name}")
            print(f"Gradient Checkpointing: {self.use_gradient_checkpointing}")
        print(f"Sequence Length: {self.model_config.seq_length}")
        print(f"\nMemory Breakdown:")
        print(f"{'-'*70}")
        
        for component, size_gb in memory.items():
            if component == "Total (GB)":
                print(f"{'-'*70}")
                print(f"{component:30s}: {size_gb:>10.2f} GB")
            else:
                print(f"{component:30s}: {size_gb:>10.2f} GB")
        
        print(f"{'='*70}\n")
        
        # Calculate number of GPUs needed
        num_gpus_a100_80 = memory["Total (GB)"] / A100_80GB
        num_gpus_h100_80 = memory["Total (GB)"] / H100_80GB
        num_gpus_a100_40 = memory["Total (GB)"] / A100_40GB
        print(f"Estimated GPUs needed (A100 80GB): {num_gpus_a100_80:.1f}")
        print(f"Estimated GPUs needed (H100 80GB): {num_gpus_h100_80:.1f}")
        print(f"Estimated GPUs needed (A100 40GB): {num_gpus_a100_40:.1f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate GPU memory requirements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use HuggingFace model name
  python gpu_memory_calculator.py --model meta-llama/Meta-Llama-3-8B --mode training
  
  # Use legacy short name (will be converted to HuggingFace name)
  python gpu_memory_calculator.py --model llama3-8b --mode training
  
  # Any HuggingFace model
  python gpu_memory_calculator.py --model microsoft/phi-2 --mode inference
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",  # Use publicly available model as default
        help="Model name (HuggingFace model identifier or legacy short name like 'llama3-8b')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "inference"],
        help="Training or inference mode"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        help="Model precision"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Sequence length (overrides model default)"
    )
    # Build optimizer choices list
    optimizer_choices = [opt.name.lower().replace("_", "-") for opt in OptimizerType]
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=optimizer_choices,
        help="Optimizer type"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing"
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV cache for inference"
    )
    parser.add_argument(
        "--load-full-model",
        action="store_true",
        help="Load full model to count parameters exactly (uses significant memory). "
             "By default, parameters are estimated from architecture to save memory."
    )
    
    args = parser.parse_args()
    
    # Get model config from transformers
    model_name = args.model
    # Check if it's a legacy short name
    if model_name in LEGACY_MODEL_NAMES:
        model_name = LEGACY_MODEL_NAMES[model_name]
        print(f"Using legacy model name, converted to: {model_name}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library is required to load model configurations.")
        print("Install with: pip install transformers")
        return
    
    try:
        model_config = load_model_config_from_transformers(model_name, load_full_model=args.load_full_model)
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        return
    
    if args.seq_length:
        model_config.seq_length = args.seq_length
    
    # Parse precision
    precision_map = {
        "fp32": Precision.FP32,
        "fp16": Precision.FP16,
        "bf16": Precision.BF16,
        "int8": Precision.INT8,
        "int4": Precision.INT4,
    }
    precision = precision_map[args.precision]
    
    # Parse optimizer (handle both dash and underscore formats)
    optimizer_name = args.optimizer.replace("-", "_")
    optimizer_map = {opt.name.lower(): opt for opt in OptimizerType}
    optimizer = optimizer_map[optimizer_name]
    
    # Create calculator
    calculator = GPUMemoryCalculator(
        model_config=model_config,
        precision=precision,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimizer=optimizer,
        use_gradient_checkpointing=args.gradient_checkpointing,
        kv_cache_enabled=not args.no_kv_cache,
    )
    
    # Print breakdown
    calculator.print_memory_breakdown(mode=args.mode)


if __name__ == "__main__":
    main()
