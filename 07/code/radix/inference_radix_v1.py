"""
RadixAttention inference implementation.

This script implements inference using RadixAttention for prefix cache reuse,
similar to SGLang's approach.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict
import time

try:
    from .radix_cache_v1 import RadixCache, RadixKey, MatchResult
    from .radix_attention import RadixAttention
except ImportError:
    from radix_cache_v1 import RadixCache, RadixKey, MatchResult
    from radix_attention import RadixAttention


class RadixAttentionModelWrapper:
    """
    Model wrapper using RadixAttention for prefix cache reuse.
    
    This implementation:
    - Uses RadixCache to store and share KV cache prefixes
    - Matches prefixes across requests to reuse cached values
    - Only computes KV cache for new tokens not in cache
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        page_size: int = 1,
    ):
        """
        Initialize the RadixAttention model wrapper.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            page_size: Page size for cache (1 = token-level)
        """
        self.device = device
        self.page_size = page_size
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device
        )
        self.model.eval()
        
        # Get model config
        config = self.model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        
        print(f"Model config: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, "
              f"{self.head_dim} head_dim, {self.num_layers} layers")
        print(f"Using RadixAttention with page_size={page_size}")
        
        # Create RadixAttention for each layer
        self.radix_attentions = [
            RadixAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                layer_id=i,
                device=device,
                page_size=page_size,
            )
            for i in range(self.num_layers)
        ]
        
        # Track sequences
        self.sequences: Dict[int, Dict] = {}
        self.next_seq_id = 0
        
        # Per-sequence KV cache (like baseline, but we'll add radix sharing)
        # Format: {seq_id: {layer_idx: {'k': tensor, 'v': tensor}}}
        self.kv_caches: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        
        # Global KV cache storage for radix tree (shared prefixes)
        # Format: {layer_idx: {cache_idx: {'k': tensor, 'v': tensor}}}
        self.global_kv_cache: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {
            i: {} for i in range(self.num_layers)
        }
        self.next_cache_idx = 0
    
    def _get_attention_layer(self, layer_idx: int):
        """Get the attention layer from the model."""
        return self.model.model.layers[layer_idx].self_attn
    
    def _allocate_cache_slot(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> int:
        """Allocate a new cache slot and store K/V."""
        cache_idx = self.next_cache_idx
        self.next_cache_idx += 1
        
        # Store in global cache
        self.global_kv_cache[layer_idx][cache_idx] = {
            'k': k.clone(),
            'v': v.clone(),
        }
        
        return cache_idx
    
    def _get_cached_kv(self, layer_idx: int, cache_indices: torch.Tensor) -> tuple:
        """Retrieve cached K/V from indices."""
        if len(cache_indices) == 0:
            return None, None
        
        # Get cached K/V tensors
        cached_k_list = []
        cached_v_list = []
        
        for idx in cache_indices.cpu().tolist():
            if idx in self.global_kv_cache[layer_idx]:
                cached_k_list.append(self.global_kv_cache[layer_idx][idx]['k'])
                cached_v_list.append(self.global_kv_cache[layer_idx][idx]['v'])
        
        if not cached_k_list:
            return None, None
        
        # Concatenate: [num_heads, cached_len, head_dim]
        cached_k = torch.cat(cached_k_list, dim=1)
        cached_v = torch.cat(cached_v_list, dim=1)
        
        return cached_k, cached_v
    
    def prefill(self, prompt: str, seq_id: Optional[int] = None) -> int:
        """
        Process the prompt and cache KV using RadixAttention.
        
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
            "total_tokens": len(prompt_tokens),
        }
        
        # Initialize KV cache for this sequence
        if seq_id not in self.kv_caches:
            self.kv_caches[seq_id] = {}
        
        print(f"\n[Prefill] Sequence {seq_id}: Processing {len(prompt_tokens)} prompt tokens")
        
        # Try to match prefix in cache
        match_result = self.radix_attentions[0].match_prefix(seq_id, prompt_tokens)
        matched_len = len(match_result.device_indices)
        
        if matched_len > 0:
            print(f"[Prefill] Sequence {seq_id}: Found {matched_len} cached prefix tokens")
        
        # Process tokens using model
        with torch.no_grad():
            # If we have a cache hit, we need to process only new tokens
            if matched_len > 0:
                # For simplicity, we'll process all tokens but use cached KV where available
                # In a full implementation, we'd skip computation for cached tokens
                outputs = self.model(input_ids=tokens, use_cache=True)
            else:
                outputs = self.model(input_ids=tokens, use_cache=True)
            
            past_key_values = outputs.past_key_values
            
            # Extract and store KV cache, checking for prefix reuse
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
                
                # Convert to [num_heads, seq_len, head_dim] for storage
                k_cache = k[0]  # [num_kv_heads, seq_len, head_dim]
                v_cache = v[0]
                
                # Handle GQA: repeat K and V if needed
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_cache = k_cache.repeat_interleave(repeat_factor, dim=0)
                    v_cache = v_cache.repeat_interleave(repeat_factor, dim=0)
                
                # Store in per-sequence cache (like baseline) - THIS IS CRITICAL
                self.kv_caches[seq_id][layer_idx] = {
                    'k': k_cache.clone(),
                    'v': v_cache.clone()
                }
                
                # Also store KV cache indices in radix tree for prefix sharing
                # Create indices for each token position
                kv_indices = []
                for pos in range(len(prompt_tokens)):
                    cache_idx = self._allocate_cache_slot(layer_idx, k_cache[:, pos:pos+1, :], v_cache[:, pos:pos+1, :])
                    kv_indices.append(torch.tensor([cache_idx], device=self.device, dtype=torch.int64))
                
                kv_indices_tensor = torch.cat(kv_indices)
                
                # Insert into radix cache for future prefix matching
                self.radix_attentions[layer_idx].insert_prefix(
                    seq_id, prompt_tokens, kv_indices_tensor
                )
        
        # Check what the first token would be
        logits_check = outputs.logits[:, -1, :]
        first_token_id = torch.argmax(logits_check, dim=-1).item()
        first_token_text = self.tokenizer.decode([first_token_id])
        print(f"[Prefill] Sequence {seq_id}: Cached KV for {len(prompt_tokens)} tokens")
        print(f"[Prefill] First token would be: id={first_token_id}, text='{first_token_text}'")
        
        return seq_id
    
    def decode_step(self, seq_id: int) -> Optional[int]:
        """
        Generate one token using RadixAttention with cached KV.
        
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
            
            # Process through each layer
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attention = self._get_attention_layer(layer_idx)
                
                # Apply layer norm before attention
                hidden_states_norm = layer.input_layernorm(hidden_states)
                
                # Compute Q, K, V for the current token
                q_proj = attention.q_proj(hidden_states_norm)
                k_proj = attention.k_proj(hidden_states_norm)
                v_proj = attention.v_proj(hidden_states_norm)
                
                # Reshape
                q = q_proj.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [1, num_heads, 1, head_dim]
                k = k_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
                v = v_proj.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
                
                # Get cached K and V from per-sequence cache (like baseline)
                k_cached = self.kv_caches[seq_id][layer_idx]['k']  # [num_heads, cached_len, head_dim]
                v_cached = self.kv_caches[seq_id][layer_idx]['v']  # [num_heads, cached_len, head_dim]
                
                # Handle GQA for current token's K/V
                k_new = k[0]  # [num_kv_heads, 1, head_dim]
                v_new = v[0]  # [num_kv_heads, 1, head_dim]
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_new = k_new.repeat_interleave(repeat_factor, dim=0)  # [num_heads, 1, head_dim]
                    v_new = v_new.repeat_interleave(repeat_factor, dim=0)
                
                # Reshape for batched attention computation
                k_cached_batched = k_cached.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
                v_cached_batched = v_cached.unsqueeze(0)  # [1, num_heads, cached_len, head_dim]
                
                # Compute attention over OLD cached KV (before adding current token's K/V)
                # q: [1, num_heads, 1, head_dim]
                # k_cached_batched: [1, num_heads, cached_len, head_dim]
                scale = 1.0 / (self.head_dim ** 0.5)
                scores = torch.matmul(q, k_cached_batched.transpose(-2, -1)) * scale
                attn_weights = F.softmax(scores, dim=-1)  # [1, num_heads, 1, cached_len]
                
                # Compute attention output: attn_weights @ V
                attn_output = torch.matmul(attn_weights, v_cached_batched)  # [1, num_heads, 1, head_dim]
                
                # NOW add new K, V to cache (for next step)
                k_cached = torch.cat([k_cached, k_new], dim=1)  # [num_heads, cached_len+1, head_dim]
                v_cached = torch.cat([v_cached, v_new], dim=1)  # [num_heads, cached_len+1, head_dim]
                
                # Update cache
                self.kv_caches[seq_id][layer_idx]['k'] = k_cached
                self.kv_caches[seq_id][layer_idx]['v'] = v_cached
                
                # Reshape for output projection
                attn_output = attn_output.transpose(1, 2).contiguous()  # [1, 1, num_heads, head_dim]
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
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            # Debug: print first few tokens
            if len(seq_info["generated_tokens"]) < 5:
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(f"  Debug token {len(seq_info['generated_tokens'])+1}: id={next_token_id}, text='{token_text}', logit_max={next_token_logits.max().item():.2f}")
            
            # Update sequence info
            seq_info["generated_tokens"].append(next_token_id)
            seq_info["total_tokens"] += 1
        
        return next_token_id
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text using RadixAttention.
        
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
                print(f"  Step {step + 1}: Generated {len(generated_tokens)} tokens")
        
        # Decode tokens to text
        full_tokens = self.sequences[seq_id]["prompt_tokens"] + generated_tokens
        generated_text = self.tokenizer.decode(full_tokens, skip_special_tokens=True)
        
        # Clean up
        del self.sequences[seq_id]
        
        return generated_text
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        total_kv_elements = 0
        total_sequences = len(self.sequences)
        
        for layer_idx in range(self.num_layers):
            for cache_idx, cache in self.global_kv_cache[layer_idx].items():
                k_shape = cache['k'].shape
                v_shape = cache['v'].shape
                total_kv_elements += k_shape[0] * k_shape[1] * k_shape[2]  # K cache
                total_kv_elements += v_shape[0] * v_shape[1] * v_shape[2]  # V cache
        
        total_memory_mb = (total_kv_elements * 2) / (1024 * 1024)  # 2 bytes per float16
        
        return {
            "total_sequences": total_sequences,
            "total_kv_elements": total_kv_elements,
            "total_memory_mb": total_memory_mb
        }


def main():
    """Main function to run RadixAttention inference."""
    print("=" * 60)
    print("RadixAttention Inference (Prefix Cache Reuse)")
    print("=" * 60)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = RadixAttentionModelWrapper(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
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
        
        start_time = time.time()
        generated = model_wrapper.generate(prompt, max_new_tokens=50)
        elapsed_time = time.time() - start_time
        
        print(f"\nGenerated text:")
        print(generated)
        print(f"\nTime taken: {elapsed_time:.2f} seconds")
        print()
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("Final Memory Stats:")
    print(f"{'=' * 60}")
    stats = model_wrapper.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
