"""
Inference script using custom PagedAttention for KV cache management.

This script demonstrates how to use the custom PagedAttention implementation
to manage KV cache when doing inference with Qwen/Qwen2.5-0.5B-Instruct.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Tuple
import sys
import os

# Add parent directory to path to import pa module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pa import PagedAttention

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class PagedAttentionModelWrapper:
    """
    Wrapper around a HuggingFace model to use custom PagedAttention for KV cache.
    
    This is a simplified implementation that demonstrates the concept.
    In production systems like vLLM, this is integrated at the kernel level.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        block_size: int = 16,
        device: str = "cuda"
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model name
            block_size: Block size for PagedAttention
            device: Device to use
        """
        self.device = device
        self.block_size = block_size
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device
        )
        
        # Get model config
        config = self.model.config
        self.num_heads = config.num_attention_heads
        # Qwen2.5 uses GQA, so K/V might have fewer heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        
        print(f"Model config: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, {self.head_dim} head_dim, {self.num_layers} layers")
        
        # Initialize PagedAttention for each layer
        self.paged_attentions = [
            PagedAttention(
                block_size=block_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=device
            )
            for _ in range(self.num_layers)
        ]
        
        # Track sequences
        self.sequences: dict[int, dict] = {}  # seq_id -> {prompt_tokens, generated_tokens, ...}
        self.next_seq_id = 0
    
    def _get_attention_layer(self, layer_idx: int):
        """Get the attention layer from the model."""
        return self.model.model.layers[layer_idx].self_attn
    
    def _apply_rope_hf(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE using HuggingFace's Qwen2Model implementation.
        
        This reuses the exact same RoPE logic as Qwen2Model, ensuring correctness.
        The rotary_emb is at model.model.rotary_emb, not on individual attention layers.
        
        Args:
            query_states: Query tensor of shape [B, Hq, q_len, D]
            key_states: Key tensor of shape [B, Hkv, q_len, D]
            position_ids: Position IDs of shape [B, q_len]
            
        Returns:
            Tuple of (query_states_rope, key_states_rope) with RoPE applied
        """
        # Get cos, sin from the model's rotary_emb module
        # rotary_emb is at model.model.rotary_emb (shared across all layers)
        # It expects (hidden_states, position_ids) and returns (cos, sin)
        # We use key_states as a dummy tensor for dtype/device
        cos, sin = self.model.model.rotary_emb(key_states, position_ids)
        
        # Apply RoPE using HuggingFace's helper function
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        
        return query_states, key_states
    
    def prefill(self, prompt: str, seq_id: Optional[int] = None) -> int:
        """
        Process the prompt and cache KV using PagedAttention.
        
        Args:
            prompt: Input prompt text
            seq_id: Optional sequence ID (if None, creates new sequence)
            
        Returns:
            Sequence ID
        """
        if seq_id is None:
            seq_id = self.next_seq_id
            self.next_seq_id += 1
        
        # Apply chat template if available (for Qwen models)
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Format as chat messages
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        else:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = tokens[0].tolist()
        
        # Store sequence info
        self.sequences[seq_id] = {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": [],
            "total_tokens": len(prompt_tokens)
        }
        
        print(f"\n[Prefill] Sequence {seq_id}: Processing {len(prompt_tokens)} prompt tokens")
        
        # Process prompt tokens using model's forward (includes RoPE automatically)
        with torch.no_grad():
            # Use model's forward with use_cache to get KV cache with RoPE applied
            outputs = self.model(input_ids=tokens, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Extract KV cache from past_key_values and store in PagedAttention blocks
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
                k_cache = k[0]  # [num_kv_heads, seq_len, head_dim]
                v_cache = v[0]  # [num_kv_heads, seq_len, head_dim]
                
                # Store each token's K/V in PagedAttention blocks
                for token_idx in range(k_cache.shape[1]):  # seq_len
                    k_token = k_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    v_token = v_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    
                    # Handle GQA: repeat KV heads to match Q heads
                    if self.num_kv_heads < self.num_heads:
                        repeat_factor = self.num_heads // self.num_kv_heads
                        k_token = k_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                        v_token = v_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                    
                    self.paged_attentions[layer_idx].append_kv(
                        seq_id, k_token, v_token, token_idx
                    )
        
        # Check what the first token would be
        logits_check = outputs.logits[:, -1, :]
        first_token_id = torch.argmax(logits_check, dim=-1).item()
        first_token_text = self.tokenizer.decode([first_token_id])
        print(f"[Prefill] Sequence {seq_id}: Cached KV for {len(prompt_tokens)} tokens")
        stats = self.paged_attentions[0].get_stats()
        print(f"[Prefill] Block stats: {stats}")
        print(f"[Prefill] First token would be: id={first_token_id}, text='{first_token_text}'")
        
        return seq_id
    
    def decode_step(self, seq_id: int) -> Optional[int]:
        """
        Generate one token using PagedAttention for attention computation.
        
        This implementation bypasses HuggingFace's attention and uses PagedAttention
        directly, making PA truly participate in the compute path.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Generated token ID, or None if sequence not found
        """
        if seq_id not in self.sequences:
            return None
        
        seq_info = self.sequences[seq_id]
        
        # Get the last generated token (or last prompt token if no generation yet)
        if seq_info["generated_tokens"]:
            last_token_id = seq_info["generated_tokens"][-1]
        else:
            last_token_id = seq_info["prompt_tokens"][-1]
        
        token_tensor = torch.tensor([[last_token_id]], device=self.device)
        
        with torch.no_grad():
            # Get embedding for the current token
            hidden_states = self.model.model.embed_tokens(token_tensor)
            
            # Get position for RoPE (current token position)
            current_pos = seq_info["total_tokens"]
            position_ids = torch.tensor([[current_pos]], device=self.device, dtype=torch.long)
            
            # Process through each layer manually
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attention = self._get_attention_layer(layer_idx)
                
                # Apply layer norm before attention
                hidden_states_norm = layer.input_layernorm(hidden_states)
                
                # Compute Q, K, V for the current token
                q_proj = attention.q_proj(hidden_states_norm)
                k_proj = attention.k_proj(hidden_states_norm)
                v_proj = attention.v_proj(hidden_states_norm)
                
                # Reshape Q, K, V to [B, H, seq_len, D] format for RoPE
                # Q: [1, 1, num_heads * head_dim] -> [1, num_heads, 1, head_dim]
                q = q_proj.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
                # K, V: [1, 1, num_kv_heads * head_dim] -> [1, num_kv_heads, 1, head_dim]
                k = k_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # Apply RoPE using HuggingFace's implementation
                q_rope, k_rope = self._apply_rope_hf(
                    q, k, position_ids
                )
                # q_rope: [1, num_heads, 1, head_dim]
                # k_rope: [1, num_kv_heads, 1, head_dim]
                
                # Extract the single token's Q, K, V (remove batch and seq dims)
                q_rope_token = q_rope[0, :, 0, :]  # [num_heads, head_dim]
                k_rope_token = k_rope[0, :, 0, :]  # [num_kv_heads, head_dim]
                v_token = v[0, :, 0, :]  # [num_kv_heads, head_dim] (V doesn't need RoPE)
                
                # Handle GQA: repeat KV heads to match Q heads for storage/computation
                # Note: According to a3.md, PA should store KV in Hkv heads and map q heads to kv heads
                # inside PA kernel. However, our current PA implementation expects [num_heads, head_dim]
                # for both Q and K/V, so we repeat here for compatibility.
                # TODO: Refactor PA to handle GQA natively without repetition
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_rope_token = k_rope_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                    v_token = v_token.repeat_interleave(repeat_factor, dim=0)  # [num_heads, head_dim]
                
                # Use PagedAttention to compute attention over cached KV
                # This is the key: PA computes attention, not HF
                # The cached K/V already have RoPE applied (from prefill)
                # The new Q has RoPE applied above
                attn_output_paged = self.paged_attentions[layer_idx].compute_attention(
                    seq_id, q_rope_token
                )
                
                # Cache K and V for the current token (for next step)
                # Store with RoPE applied to match prefill behavior
                token_idx = seq_info["total_tokens"]
                self.paged_attentions[layer_idx].append_kv(seq_id, k_rope_token, v_token, token_idx)
                
                # Reshape for output projection
                attn_output = attn_output_paged.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(1, 1, -1)
                attn_output = attention.o_proj(attn_output)
                
                # Residual connection
                hidden_states = hidden_states + attn_output
                
                # Feedforward
                hidden_states_norm = layer.post_attention_layernorm(hidden_states)
                mlp_output = layer.mlp(hidden_states_norm)
                hidden_states = hidden_states + mlp_output
            
            # Apply final layer norm
            hidden_states = self.model.model.norm(hidden_states)
            
            # Get logits and sample next token
            logits = self.model.lm_head(hidden_states)
            
            # Apply temperature and sample (or use argmax for deterministic)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Debug: print first few tokens
            if len(seq_info["generated_tokens"]) < 3:
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(f"  Debug: Generated token {len(seq_info['generated_tokens'])+1}: id={next_token_id}, text='{token_text}'")
            
            # Update sequence info
            seq_info["generated_tokens"].append(next_token_id)
            seq_info["total_tokens"] += 1
        
        return next_token_id
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using PagedAttention.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        # Prefill
        seq_id = self.prefill(prompt)
        
        # Decode
        print(f"\n[Decode] Sequence {seq_id}: Generating up to {max_new_tokens} tokens")
        generated_tokens = []
        
        for step in range(max_new_tokens):
            token_id = self.decode_step(seq_id)
            if token_id is None:
                break
            
            generated_tokens.append(token_id)
            
            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break
            
            if (step + 1) % 10 == 0:
                stats = self.paged_attentions[0].get_stats()
                print(f"  Step {step + 1}: {stats['total_tokens']} total tokens, "
                      f"{stats['allocated_blocks']} blocks allocated")
        
        # Decode tokens to text
        full_tokens = self.sequences[seq_id]["prompt_tokens"] + generated_tokens
        generated_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        # Clean up
        for layer_idx in range(self.num_layers):
            self.paged_attentions[layer_idx].free_sequence(seq_id)
        del self.sequences[seq_id]
        
        return generated_text


def main():
    """Main function to run inference with PagedAttention."""
    print("=" * 60)
    print("PagedAttention Inference Demo")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = PagedAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        block_size=16,
        device=device
    )
    
    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Prompt {i + 1}: {prompt}")
        print(f"{'=' * 60}")
        
        generated = model_wrapper.generate(prompt, max_new_tokens=50)
        
        print(f"\nGenerated text:")
        print(generated)
        print()
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("Final Block Manager Stats:")
    print(f"{'=' * 60}")
    stats = model_wrapper.paged_attentions[0].get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
