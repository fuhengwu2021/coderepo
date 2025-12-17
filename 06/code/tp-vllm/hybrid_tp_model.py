"""
Hybrid TP Model: Mix TP layers with HuggingFace layers

This module implements a hybrid approach:
- Attention and MLP layers use TP (our implementation)
- Embedding, LayerNorm, and LM Head use HuggingFace original
- This allows actual text generation while demonstrating TP
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, List, Dict
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

try:
    from .attention import TensorParallelAttention
    from .mlp import TensorParallelMLP
    from .model_wrapper import TPModelWrapper
    from .parallel_state import get_tensor_model_parallel_rank
except ImportError:
    from attention import TensorParallelAttention
    from mlp import TensorParallelMLP
    from model_wrapper import TPModelWrapper
    from parallel_state import get_tensor_model_parallel_rank


class HybridTPDecoderLayer(nn.Module):
    """Single decoder layer with TP attention and MLP, HF norm"""
    
    def __init__(
        self,
        hf_layer,  # Original HF layer for norm and reference
        tp_attention: TensorParallelAttention,
        tp_mlp: TensorParallelMLP,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        
        # Use HF's layer norm (not sharded)
        # Note: These will be moved to device in HybridTPModel.__init__
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        
        # Use our TP layers
        self.self_attn = tp_attention
        self.mlp = tp_mlp
        
        # Get RoPE from HF model (Qwen2.5 has rotary_emb in model.model, not in each layer)
        # Qwen models require RoPE for correct positional encoding
        # Note: We'll get rotary_emb from the parent model, not from hf_layer
        # This will be set in HybridTPModel.__init__
        self.rotary_emb = None  # Will be set by parent model
    
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using HuggingFace's implementation"""
        if self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(k, position_ids)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                return q_rope, k_rope
            except Exception as e:
                # Fallback if RoPE fails
                return q, k
        return q, k
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through decoder layer
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len] position indices
            kv_cache: Optional (k_cache, v_cache) tuple
        
        Returns:
            (output, (k_cache, v_cache))
        """
        # Residual connection
        residual = hidden_states
        
        # Input layer norm (HF)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Attention (TP)
        # Apply RoPE if available (critical for Qwen models!)
        rotary_emb = getattr(self, 'rotary_emb', None)
        attn_output, kv_cache = self.self_attn(
            hidden_states,
            kv_cache=kv_cache,
            position_ids=position_ids,
            rotary_emb=rotary_emb,
        )
        
        # Residual connection
        hidden_states = residual + attn_output
        
        # Residual for MLP
        residual = hidden_states
        
        # Post-attention layer norm (HF)
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # MLP (TP)
        mlp_output = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + mlp_output
        
        return hidden_states, kv_cache


class HybridTPModel(nn.Module):
    """
    Hybrid TP Model: Mix TP layers with HuggingFace layers
    
    - Embedding: HF original
    - Transformer layers: TP attention + TP MLP, HF norm
    - Final norm: HF original
    - LM Head: HF original (each rank has full vocab for simplicity)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.tp_rank = get_tensor_model_parallel_rank()
        
        if self.tp_rank == 0:
            print(f"Creating HybridTPModel from {model_name}...")
        
        # Handle device string or device object
        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device
        self.device = device_obj
        
        # Load HF model for reference and non-TP layers
        dtype = dtype or torch.float16 if str(device_obj).startswith("cuda") else torch.float32
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="cpu",  # Load on CPU first, then move to device
        )
        
        # Get config
        self.config = hf_model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = getattr(self.config, 'intermediate_size', 4 * self.hidden_size)
        
        # Handle device string or device object
        if isinstance(device, str):
            device_obj = torch.device(device)
        else:
            device_obj = device
        self.device = device_obj
        
        # Use HF's embedding (not sharded for simplicity)
        self.embed_tokens = hf_model.model.embed_tokens
        self.embed_tokens = self.embed_tokens.to(device_obj)
        
        # Use HF's final norm
        self.norm = hf_model.model.norm
        self.norm = self.norm.to(device_obj)
        
        # Use HF's LM head (each rank has full vocab for simplicity)
        # In production, would use VocabParallelEmbedding
        self.lm_head = hf_model.lm_head
        self.lm_head = self.lm_head.to(device_obj)
        
        # Get RoPE from HF model (Qwen2.5 has rotary_emb in model.model, shared across all layers)
        # This is CRITICAL for Qwen models!
        if hasattr(hf_model.model, 'rotary_emb'):
            self.rotary_emb = hf_model.model.rotary_emb
            self.rotary_emb = self.rotary_emb.to(device_obj)
            if self.tp_rank == 0:
                print(f"  Found rotary_emb in model.model (shared across layers)")
        else:
            self.rotary_emb = None
            if self.tp_rank == 0:
                print("  Warning: No rotary_emb found in model.model - this may cause incorrect results!")
        
        # Create TP wrapper for weight loading (use device_obj as string for compatibility)
        tp_wrapper = TPModelWrapper(model_name=model_name, device=str(device_obj), dtype=dtype)
        
        # Create hybrid decoder layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            hf_layer = hf_model.model.layers[i]
            
            # Create TP attention and MLP
            tp_attention = tp_wrapper.create_tp_attention_layer(i)
            tp_mlp = tp_wrapper.create_tp_mlp_layer(i)
            
            # Create hybrid layer
            hybrid_layer = HybridTPDecoderLayer(
                hf_layer=hf_layer,
                tp_attention=tp_attention,
                tp_mlp=tp_mlp,
                device=device_obj,
            )
            # Set rotary_emb from parent model (shared across all layers for Qwen)
            hybrid_layer.rotary_emb = self.rotary_emb
            
            # Ensure all components are on the correct device
            # LayerNorm needs to be moved explicitly (they come from HF model which is on CPU)
            hybrid_layer.input_layernorm = hybrid_layer.input_layernorm.to(device_obj)
            hybrid_layer.post_attention_layernorm = hybrid_layer.post_attention_layernorm.to(device_obj)
            # Move the entire layer to device to ensure all submodules are on device
            hybrid_layer = hybrid_layer.to(device_obj)
            self.layers.append(hybrid_layer)
        
        # Clean up HF model (we've extracted what we need)
        del hf_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.eval()
        
        if self.tp_rank == 0:
            print(f"HybridTPModel created: {self.num_layers} layers with TP attention/MLP")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through hybrid TP model
        
        Args:
            input_ids: [batch, seq_len] token ids
            position_ids: [batch, seq_len] position indices
            kv_caches: Optional list of (k_cache, v_cache) for each layer
        
        Returns:
            (logits, kv_caches): logits [batch, seq_len, vocab_size], updated kv_caches
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding (HF)
        hidden_states = self.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        
        # Create position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Initialize KV caches if not provided
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            hidden_states, kv_cache = layer(
                hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
            )
            kv_caches[i] = kv_cache
            # Add progress output for debugging (only on rank 0, every 6 layers to reduce overhead)
            if self.tp_rank == 0 and (i % 6 == 0 or i == self.num_layers - 1):
                if i == self.num_layers - 1:
                    print(f"    Layer {i+1}/{self.num_layers} done")
                elif i == 0:
                    print(f"    Layer {i+1}/{self.num_layers}...", end='', flush=True)
                else:
                    print(f" {i+1}/{self.num_layers}...", end='', flush=True)
        
        # Final norm (HF)
        hidden_states = self.norm(hidden_states)
        
        # LM Head (HF - each rank has full vocab)
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        return logits, kv_caches
    
    def generate(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate text from prompt using KV cache for efficiency
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Generated text
        """
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        else:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        input_ids = input_ids.to(self.device)
        generated_ids = input_ids.clone()
        
        # Prefill: process all prompt tokens
        # Note: On CPU this will be slow, but it will work
        rank = get_tensor_model_parallel_rank()
        
        with torch.inference_mode():
            if rank == 0:
                print(f"  Prefill: processing {input_ids.shape[1]} prompt tokens through {self.num_layers} layers...")
            
            # Prefill: forward through model with kv_caches=None
            # This will return both logits and kv_caches
            import time
            start_time = time.time()
            
            try:
                logits, kv_caches = self.forward(input_ids, kv_caches=None)  # [batch, seq_len, vocab_size]
            except Exception as e:
                if rank == 0:
                    print(f"\n  Error during forward pass: {e}")
                    import traceback
                    traceback.print_exc()
                raise
            
            elapsed = time.time() - start_time
            
            # Synchronize all ranks before printing
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
            
            if rank == 0:
                print(f"\n  Prefill completed in {elapsed:.2f}s")
            
            # Get next token from last position
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            if rank == 0:
                print(f"  Generating {max_new_tokens} tokens with KV cache incremental decode...")
            
            # Generate tokens one by one using KV cache incremental decode
            # This is the correct way: only forward the new token, reuse KV cache
            for step in range(max_new_tokens):
                if rank == 0 and step % 10 == 0:
                    print(f"    Step {step}/{max_new_tokens}...", end='\r', flush=True)
                
                # Sample next token
                if do_sample and temperature > 0:
                    next_token_logits_scaled = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits_scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS before appending
                if next_token.item() == tokenizer.eos_token_id:
                    if rank == 0:
                        print(f"    EOS token reached at step {step}")
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Incremental decode: only forward the new token with KV cache
                # Position is the current absolute position (after appending)
                current_position = generated_ids.shape[1] - 1
                position_ids = torch.tensor([[current_position]], device=self.device, dtype=torch.long)
                
                # Forward only the new token, reuse kv_caches
                new_token_ids = next_token.unsqueeze(0)  # [batch, 1]
                logits, kv_caches = self.forward(new_token_ids, position_ids=position_ids, kv_caches=kv_caches)
                next_token_logits = logits[0, -1, :]  # [vocab_size]
            
            if rank == 0:
                print()  # New line after progress
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # If chat template was used, extract only the assistant's response
        # The chat template adds system/user/assistant prefixes
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            # Try to extract just the assistant response
            # Look for "assistant" marker
            if "assistant" in generated_text.lower():
                parts = generated_text.split("assistant")
                if len(parts) > 1:
                    # Take everything after "assistant"
                    assistant_response = parts[-1].strip()
                    # Remove any remaining template artifacts
                    if assistant_response:
                        generated_text = assistant_response
        
        return generated_text
