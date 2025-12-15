"""
Simple test to verify tensor parallelism correctness
This can be run without distributed setup to test the logic
"""
import torch
import torch.nn as nn


def test_column_parallel_math():
    """Test that column parallel produces correct results"""
    print("Testing Column Parallel Math...")
    
    input_size = 8
    output_size = 16
    batch_size = 2
    tp_size = 2
    
    # Create full weight matrix
    full_weight = torch.randn(input_size, output_size)
    full_bias = torch.randn(output_size)
    
    # Split weight along columns
    weight_shard_0 = full_weight[:, :output_size//tp_size]
    weight_shard_1 = full_weight[:, output_size//tp_size:]
    bias_shard_0 = full_bias[:output_size//tp_size]
    bias_shard_1 = full_bias[output_size//tp_size:]
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Compute with full matrix (F.linear expects weight to be [out_features, in_features])
    output_full = F.linear(x, full_weight.t(), full_bias)
    
    # Compute with sharded matrices
    output_shard_0 = F.linear(x, weight_shard_0.t(), bias_shard_0)
    output_shard_1 = F.linear(x, weight_shard_1.t(), bias_shard_1)
    output_sharded = torch.cat([output_shard_0, output_shard_1], dim=-1)
    
    # Verify they match
    assert torch.allclose(output_full, output_sharded), "Column parallel math incorrect!"
    print("✓ Column parallel math is correct")


def test_row_parallel_math():
    """Test that row parallel produces correct results"""
    print("Testing Row Parallel Math...")
    
    input_size = 16
    output_size = 8
    batch_size = 2
    tp_size = 2
    
    # Create full weight matrix
    full_weight = torch.randn(input_size, output_size)
    full_bias = torch.randn(output_size)
    
    # Split weight along rows and input along last dimension
    weight_shard_0 = full_weight[:input_size//tp_size, :]
    weight_shard_1 = full_weight[input_size//tp_size:, :]
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Split input along last dimension
    x_shard_0 = x[:, :input_size//tp_size]
    x_shard_1 = x[:, input_size//tp_size:]
    
    # Compute with full matrix (F.linear expects weight to be [out_features, in_features])
    output_full = F.linear(x, full_weight.t(), full_bias)
    
    # Compute with sharded matrices
    output_shard_0 = F.linear(x_shard_0, weight_shard_0.t())
    output_shard_1 = F.linear(x_shard_1, weight_shard_1.t())
    output_sharded = output_shard_0 + output_shard_1 + full_bias
    
    # Verify they match
    assert torch.allclose(output_full, output_sharded), "Row parallel math incorrect!"
    print("✓ Row parallel math is correct")


def test_mlp_math():
    """Test that MLP with TP produces correct results"""
    print("Testing MLP Math...")
    
    hidden_size = 8
    intermediate_size = 16
    batch_size = 2
    seq_len = 3
    tp_size = 2
    
    # Create full weight matrices
    gate_weight_full = torch.randn(hidden_size, intermediate_size)
    up_weight_full = torch.randn(hidden_size, intermediate_size)
    down_weight_full = torch.randn(intermediate_size, hidden_size)
    
    # Split for column parallel (gate and up)
    gate_weight_0 = gate_weight_full[:, :intermediate_size//tp_size]
    gate_weight_1 = gate_weight_full[:, intermediate_size//tp_size:]
    up_weight_0 = up_weight_full[:, :intermediate_size//tp_size]
    up_weight_1 = up_weight_full[:, intermediate_size//tp_size:]
    
    # Split for row parallel (down)
    down_weight_0 = down_weight_full[:intermediate_size//tp_size, :]
    down_weight_1 = down_weight_full[intermediate_size//tp_size:, :]
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass with full matrices (F.linear expects weight to be [out_features, in_features])
    gate_full = F.linear(x, gate_weight_full.t())
    up_full = F.linear(x, up_weight_full.t())
    activated_full = F.silu(gate_full) * up_full
    output_full = F.linear(activated_full, down_weight_full.t())
    
    # Forward pass with sharded matrices
    # Column parallel: gate and up
    gate_0 = F.linear(x, gate_weight_0.t())
    gate_1 = F.linear(x, gate_weight_1.t())
    up_0 = F.linear(x, up_weight_0.t())
    up_1 = F.linear(x, up_weight_1.t())
    
    # Activation (element-wise, operates on sharded data)
    activated_0 = F.silu(gate_0) * up_0
    activated_1 = F.silu(gate_1) * up_1
    
    # Row parallel: down
    down_0 = F.linear(activated_0, down_weight_0.t())
    down_1 = F.linear(activated_1, down_weight_1.t())
    output_sharded = down_0 + down_1
    
    # Verify they match
    assert torch.allclose(output_full, output_sharded, atol=1e-5), "MLP math incorrect!"
    print("✓ MLP math is correct")


if __name__ == "__main__":
    import torch.nn.functional as F
    
    print("="*60)
    print("Tensor Parallelism Math Tests")
    print("="*60)
    
    test_column_parallel_math()
    test_row_parallel_math()
    test_mlp_math()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

