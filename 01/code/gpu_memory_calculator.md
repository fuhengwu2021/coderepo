# GPU Memory Calculator

This tool calculates GPU memory requirements for training and inference of large language models.

## Quick Start

```bash
# Install dependencies
pip install transformers

# Calculate memory for a model (uses meta device, zero memory allocation)
python gpu_memory_calculator.py --model microsoft/phi-2 --mode training

# With custom settings
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode training \
    --precision bf16 \
    --optimizer adamw \
    --batch-size 4 \
    --gradient-checkpointing
```

## Features

- **Automatic model loading**: Loads any model from HuggingFace Hub automatically
- **Zero-memory parameter counting**: Uses PyTorch meta device to count parameters without allocating memory
- **Comprehensive memory calculation**: Model weights, gradients, optimizer states, activations, KV cache
- **Multiple precisions**: FP32, FP16, BF16, INT8, INT4
- **15+ optimizers**: SGD, Adam, AdamW, Lion, Adafactor, and more
- **Training and inference modes**: Different memory calculations for each
- **Gradient checkpointing support**: Accounts for memory reduction
- **No hardcoded values**: All constants are configurable

## Requirements

- **Python**: 3.7+
- **PyTorch**: 2.0+ (for meta device support)
- **transformers**: Latest version recommended

## Installation

Install the required dependencies:

```bash
# Using pip
pip install transformers torch

# Or using conda
conda install -c conda-forge transformers pytorch
```

**Note**: The script uses only Python standard library plus `transformers`. PyTorch is required for meta device support (PyTorch 2.0+). If you have an older PyTorch version, use `--load-full-model` flag.

The script automatically loads model configurations from HuggingFace Hub, so you don't need to hardcode model parameters.

**Parameter Counting Method**:

The script uses **PyTorch meta device** by default to count parameters with zero memory allocation:

- **Zero memory allocation**: Meta device creates tensors with shape/dtype metadata only, no actual data
- **Highly accurate**: Counts actual model structure parameters (99%+ accuracy)
- **Requires PyTorch 2.0+**: Meta device support
- **No fallback**: If meta device fails, script exits with clear error message
- **Alternative**: Use `--load-full-model` to load complete model with weights (requires significant memory)

**Why Meta Device?**
- Creates model structure without allocating actual memory
- Only stores tensor shapes and dtypes, not actual data
- Perfect for parameter counting: accurate + zero memory overhead
- For 8B model: uses ~0MB vs ~32GB for full model loading

**Limitations:**
- Some operations may not support meta device (custom layers, certain initializations)
- If meta device fails, use `--load-full-model` for accurate parameter count

**Gated Models**: Some models (like Meta-Llama-3) are gated and require authorization. To access them:
1. Visit the model page on HuggingFace and request access
2. Login: `huggingface-cli login`
3. Accept the model's terms of use
4. Try again after authorization

For testing, you can use publicly available models like `microsoft/phi-2` or `mistralai/Mistral-7B-v0.1`.

## Usage

### Basic Usage

Calculate memory for a model with default settings:

```bash
# Using HuggingFace model name (default: microsoft/phi-2)
python gpu_memory_calculator.py --model microsoft/phi-2 --mode training

# Or use legacy short name (automatically converted)
python gpu_memory_calculator.py --model llama3-8b --mode training
```

### Training Examples

**Microsoft Phi-2 with AdamW optimizer, BF16 precision:**
```bash
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode training \
    --precision bf16 \
    --optimizer adamw \
    --batch-size 4 \
    --seq-length 2048
```

**With gradient checkpointing (reduces activation memory by ~70%):**
```bash
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode training \
    --precision bf16 \
    --optimizer adamw \
    --batch-size 4 \
    --gradient-checkpointing
```

**With different optimizer (SGD with momentum):**
```bash
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode training \
    --optimizer sgd-momentum \
    --precision bf16
```

### Inference Examples

**Inference with KV cache:**
```bash
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode inference \
    --precision bf16 \
    --batch-size 1 \
    --seq-length 2048
```

**Inference without KV cache:**
```bash
python gpu_memory_calculator.py \
    --model microsoft/phi-2 \
    --mode inference \
    --precision bf16 \
    --no-kv-cache
```

**Any HuggingFace model:**
```bash
# Microsoft Phi-2
python gpu_memory_calculator.py --model microsoft/phi-2 --mode inference

# Mistral
python gpu_memory_calculator.py --model mistralai/Mistral-7B-v0.1 --mode training

# Llama models (requires access authorization)
python gpu_memory_calculator.py --model meta-llama/Meta-Llama-3-8B --mode training
```

**Load full model for exact parameter count (uses significant memory):**
```bash
# By default, uses meta device (zero memory allocation)
python gpu_memory_calculator.py --model microsoft/phi-2 --mode training

# Use --load-full-model if meta device fails (requires ~2-3× model size in RAM)
python gpu_memory_calculator.py --model microsoft/phi-2 --mode training --load-full-model
```

### Model Names

The script supports any model available on HuggingFace Hub. Use the full model identifier (e.g., `meta-llama/Meta-Llama-3-8B`).

Legacy short names are also supported and automatically converted:
- `llama3-8b` → `meta-llama/Meta-Llama-3-8B`
- `llama3-70b` → `meta-llama/Meta-Llama-3-70B`
- `llama2-7b` → `meta-llama/Llama-2-7b-hf`
- `llama2-13b` → `meta-llama/Llama-2-13b-hf`

### Available Optimizers

- `sgd`: SGD (minimal memory)
- `sgd-momentum`: SGD with Momentum (1× model size)
- `nesterov`: Nesterov Accelerated Gradient (1×)
- `adagrad`: Adagrad (1×)
- `rmsprop`: RMSProp (1×)
- `adam`: Adam (2× model size)
- `adamw`: AdamW (2×)
- `adafactor`: Adafactor (0.5×, factorized)
- `lamb`: LAMB (2×)
- `lion`: Lion (1×)
- `nadam`: NAdam (2×)
- `amsgrad`: AMSGrad (2×)
- `sparse-adam`: SparseAdam (2×)
- `shampoo`: Shampoo (>2×, varies)
- `adabelief`: AdaBelief (2×)

### Available Precisions

- `fp32`: 32-bit float (4 bytes/param)
- `fp16`: 16-bit float (2 bytes/param)
- `bf16`: BFloat16 (2 bytes/param, recommended for training)
- `int8`: 8-bit integer (1 byte/param)
- `int4`: 4-bit integer (0.5 bytes/param)

## Memory Components

### Training Memory

1. **Model Weights**: Model parameters in specified precision
2. **Gradients**: Gradient values (same size as weights)
3. **Optimizer States**: Optimizer-specific states (varies by optimizer)
4. **Activations**: Intermediate layer activations (scales with batch size and sequence length)
5. **Embeddings**: Embedding layer weights
6. **Input Data**: Input batch data

### Inference Memory

1. **Model Weights**: Model parameters
2. **KV Cache**: Key-value cache for attention mechanism
3. **Embeddings**: Embedding layer weights
4. **Input Data**: Input batch data
5. **Temp Activations**: Temporary activations (much smaller than training)

## Example Output

```
Loading model configuration from HuggingFace: microsoft/phi-2
Counting parameters using meta device (zero memory allocation)...
Counted 2,779,683,840 parameters using meta device
Loaded configuration:
  Parameters: 2,779,683,840
  Hidden size: 2560
  Layers: 32
  Attention heads: 32
  Vocab size: 51200
  Intermediate size: 10240
  Max sequence length: 2048

======================================================================
GPU Memory Breakdown: phi-2
======================================================================
Mode: TRAINING
Precision: BF16
Batch Size: 4
Gradient Accumulation Steps: 1
Optimizer: ADAMW
Gradient Checkpointing: False
Sequence Length: 2048

Memory Breakdown:
----------------------------------------------------------------------
Model Weights                 :       5.18 GB
Gradients                     :       5.18 GB
Optimizer States              :      10.36 GB
Activations                   :      11.25 GB
Embeddings                    :       0.24 GB
Input Data                    :       0.04 GB
----------------------------------------------------------------------
Total (GB)                    :      32.24 GB
======================================================================

Estimated GPUs needed (A100 80GB): 0.4
Estimated GPUs needed (H100 80GB): 0.4
Estimated GPUs needed (A100 40GB): 0.8
```

## Notes

- **Parameter Counting**: Uses PyTorch meta device by default for zero-memory parameter counting. This is highly accurate (99%+) and requires no additional memory. If meta device fails, use `--load-full-model` (requires 2-3× model size in RAM).
- **Activation Memory**: Estimation is approximate and may vary based on model architecture. The calculation includes attention activations (Q, K, V, output) and FFN intermediate activations.
- **Gradient Checkpointing**: Reduces activation memory by ~70% but increases compute time (recomputes activations during backward pass).
- **KV Cache**: Memory scales linearly with batch size and sequence length. Each layer stores K and V tensors.
- **Memory Overhead**: Actual memory usage may be 10-20% higher due to framework overhead, memory fragmentation, and temporary buffers.
- **Meta Device Requirements**: Requires PyTorch 2.0+. If not available, use `--load-full-model`.
- **Gated Models**: Some models (like Meta-Llama-3) require HuggingFace authorization. See "Gated Models" section above.

## How It Works

The calculator automatically loads model configurations from HuggingFace Hub using the `transformers` library:

1. **Load Configuration**: Uses `AutoConfig.from_pretrained()` to load model configuration
2. **Extract Architecture**: Extracts parameters (hidden_size, num_layers, num_heads, intermediate_size, etc.)
3. **Count Parameters**: Uses PyTorch meta device to create model structure and count parameters
   - Creates model with `AutoModelForCausalLM.from_config()` on meta device
   - Counts all parameters: `sum(p.numel() for p in model.parameters())`
   - Zero memory allocation (only shape/dtype metadata)
4. **Calculate Memory**: Computes memory requirements for:
   - Model weights (parameters × precision)
   - Gradients (same as weights)
   - Optimizer states (varies by optimizer: 0× to 3× model size)
   - Activations (scales with batch size, sequence length, and model size)
   - KV cache (for inference, scales with batch and sequence length)

This means you can use **any model** available on HuggingFace Hub without needing to manually configure it. The script automatically extracts all necessary information from the model's configuration and structure.

### Memory Calculation Formulas

**Training Memory:**
- Model Weights: `num_params × bytes_per_param`
- Gradients: `num_params × bytes_per_param`
- Optimizer States: `num_params × bytes_per_param × optimizer_multiplier`
- Activations: `(batch_size × seq_length × hidden_size × num_layers × multiplier) × bytes_per_param`
- Embeddings: `vocab_size × hidden_size × bytes_per_param`
- Input Data: `batch_size × seq_length × hidden_size × bytes_per_param`

**Inference Memory:**
- Model Weights: `num_params × bytes_per_param`
- KV Cache: `batch_size × seq_length × hidden_size × num_layers × 2 × bytes_per_param`
- Embeddings: `vocab_size × hidden_size × bytes_per_param`
- Input Data: `batch_size × seq_length × hidden_size × bytes_per_param`
- Temp Activations: `~10% of training activations`

All constants (multipliers, ratios, GPU memory sizes) are defined at the top of the script and can be easily modified.

## Code Structure

The script is organized into several components:

- **Constants**: All hardcoded values are defined as constants at the top (GPU memory sizes, multipliers, ratios)
- **Enums**: `Precision` and `OptimizerType` define supported options
- **ModelConfig**: Dataclass to store model architecture information
- **GPUMemoryCalculator**: Main class that performs memory calculations
- **load_model_config_from_transformers**: Loads model from HuggingFace and counts parameters using meta device

### Customizing Constants

You can modify constants in the script to adjust calculations:

```python
# In gpu_memory_calculator.py
GRADIENT_CHECKPOINTING_MULTIPLIER = 0.3  # Adjust if your implementation differs
INFERENCE_ACTIVATION_RATIO = 0.1  # Adjust based on your inference setup
A100_80GB = 80.0  # Modify for different GPU configurations
```

## Troubleshooting

**Meta device not supported:**
- Upgrade to PyTorch 2.0+: `pip install --upgrade torch`
- Or use `--load-full-model` flag

**Model loading fails:**
- Check if model name is correct
- For gated models, ensure you're logged in: `huggingface-cli login`
- Check internet connection (needs to download config from HuggingFace)

**Memory calculations seem off:**
- Activation memory is an estimation and may vary
- Real-world usage includes framework overhead (10-20% more)
- Different implementations may have different memory footprints
