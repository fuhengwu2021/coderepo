"""
Simple model for Data Parallelism demo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """A simple model for demonstrating data parallelism"""
    
    def __init__(self, input_size: int = 128, hidden_size: int = 256, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """A transformer block for demonstrating data parallelism"""
    
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


class SimpleTransformer(nn.Module):
    """A simple transformer model for demonstrating data parallelism"""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_layers: int = 2, 
                 num_heads: int = 4, intermediate_size: int = 512, max_seq_len: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm and output
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

