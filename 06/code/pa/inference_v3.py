""" 
Inference script using PagedAttention v3 with Continuous Batching.

This script demonstrates continuous batching: processing multiple sequences
concurrently with dynamic scheduling (prefill + decode in same batch).
"""

import os
import sys
import time
from typing import Optional, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

# Add parent directory to path to import pa module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pa import PagedAttentionV3
from pa.scheduler import ContinuousBatchScheduler, SequenceState


class PagedAttentionModelWrapperV3:
    """
    Wrapper around a HuggingFace model to use PagedAttention v3 with continuous batching.
    
    This version supports:
    - Multiple sequences processed concurrently
    - Dynamic scheduling (prefill + decode in same batch)
    - Ragged batching (no padding)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        block_size: int = 16,
        device: str = "cuda",
        use_online_softmax: bool = True,
        max_batch_size: int = 32
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: HuggingFace model name
            block_size: Block size for PagedAttention
            device: Device to use
            use_online_softmax: Use online softmax (default) or safe_softmax
            max_batch_size: Maximum batch size for continuous batching
        """
        self.device = device
        self.block_size = block_size
        self.use_online_softmax = use_online_softmax
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        
        # Get model config
        config = self.model.config
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.num_layers = int(config.num_hidden_layers)
        
        algo_name = "Online Softmax" if use_online_softmax else "Safe Softmax"
        print(f"Model config: {self.num_heads} Q heads, {self.num_kv_heads} KV heads, "
              f"{self.head_dim} head_dim, {self.num_layers} layers")
        print(f"Using PagedAttention v3 with {algo_name} (Continuous Batching)")
        
        # Initialize scheduler
        self.scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size)
        
        # Initialize PagedAttention v3 for each layer
        self.paged_attentions = [
            PagedAttentionV3(
                block_size=block_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                device=device,
                use_online_softmax=use_online_softmax,
                num_kv_heads=self.num_kv_heads
            )
            for _ in range(self.num_layers)
        ]
    
    def _apply_rope(
        self,
        attn_module,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        kv_seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using HuggingFace's implementation."""
        try:
            # Try to get rotary_emb from model level (Qwen2 style)
            if hasattr(self.model.model, 'rotary_emb'):
                rotary_emb = self.model.model.rotary_emb
                cos, sin = rotary_emb(k, position_ids)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                return q_rope, k_rope
            # Fallback: try attention module
            elif hasattr(attn_module, 'rotary_emb'):
                rotary_emb = attn_module.rotary_emb
                cos, sin = rotary_emb(k, position_ids)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                return q_rope, k_rope
        except Exception as e:
            print(f"Warning: RoPE application failed: {e}, using identity")
            return q, k
        
        return q, k
    
    def add_request(self, prompt: str, max_new_tokens: int = 50) -> int:
        """
        Add a new request to the scheduler.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Sequence ID
        """
        # Apply chat template if available
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
        
        # Add to scheduler
        seq_id = self.scheduler.add_sequence(prompt_tokens, max_new_tokens)
        
        # Prefill: process prompt tokens and cache KV
        self._prefill_sequence(seq_id, prompt_tokens)
        
        # After prefill, move sequence to decode state
        # Note: position is already at len(prompt_tokens) after prefill
        seq_info = self.scheduler.sequences[seq_id]
        seq_info.state = SequenceState.DECODE
        # Position should already be correct, but ensure it's set
        seq_info.position = len(prompt_tokens)
        
        return seq_id
    
    def _prefill_sequence(self, seq_id: int, prompt_tokens: List[int]):
        """Prefill a sequence: process prompt and cache KV."""
        tokens_tensor = torch.tensor([prompt_tokens], device=self.device)
        
        with torch.no_grad():
            # Use model's forward to get KV cache
            outputs = self.model(input_ids=tokens_tensor, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Extract and store KV cache in PagedAttention blocks
            for layer_idx in range(self.num_layers):
                k, v = past_key_values[layer_idx]
                # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
                k_cache = k[0]  # [num_kv_heads, seq_len, head_dim]
                v_cache = v[0]  # [num_kv_heads, seq_len, head_dim]
                
                # Store each token's K/V in PagedAttention blocks
                for token_idx in range(k_cache.shape[1]):
                    k_token = k_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    v_token = v_cache[:, token_idx, :]  # [num_kv_heads, head_dim]
                    
                    self.paged_attentions[layer_idx].append_kv(
                        seq_id, k_token, v_token, token_idx
                    )
    
    def step(self) -> Tuple[List[int], List[int]]:
        """
        Process one step of continuous batching (decode only).
        
        Prefill is handled separately in add_request().
        This step only processes sequences in decode state.
        
        Returns:
            Tuple of (seq_ids, next_token_ids) for sequences that generated tokens
        """
        # Get sequences ready for decode (prefill is handled separately)
        seq_ids, positions, token_ids = self.scheduler.get_batch(
            include_prefill=False,  # Prefill already done in add_request
            include_decode=True
        )
        
        if not seq_ids:
            return [], []
        
        # Process batch (decode step)
        next_token_ids = self._process_batch(seq_ids, positions, token_ids)
        
        # Update scheduler
        self.scheduler.update_sequences(
            seq_ids,
            next_token_ids,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return seq_ids, next_token_ids
    
    def _process_batch(
        self,
        seq_ids: List[int],
        positions: List[int],
        token_ids: List[int]
    ) -> List[int]:
        """
        Process a batch of sequences (prefill or decode).
        
        Args:
            seq_ids: Sequence IDs
            positions: Position indices for each sequence
            token_ids: Token IDs to process
            
        Returns:
            List of next token IDs
        """
        batch_size = len(seq_ids)
        token_tensor = torch.tensor([token_ids], device=self.device)  # [1, batch_size]
        position_tensor = torch.tensor([positions], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            # Embed tokens
            hidden_states = self.model.model.embed_tokens(token_tensor)  # [1, batch_size, H]
            
            # Process each sequence in the batch sequentially
            # NOTE: This is pseudo-batching (sequential processing), not true ragged batching.
            # True ragged batching would flatten all tokens to T×D and process in one kernel call.
            next_token_logits_list = []
            
            for i, seq_id in enumerate(seq_ids):
                seq_hidden = hidden_states[:, i:i+1, :]  # [1, 1, H]
                seq_position = position_tensor[:, i:i+1]  # [1, 1]
                seq_pos = positions[i]
                kv_seq_len = seq_pos + 1
                
                # Process through each layer
                for layer_idx in range(self.num_layers):
                    layer = self.model.model.layers[layer_idx]
                    attn = layer.self_attn
                    
                    residual = seq_hidden
                    seq_hidden = layer.input_layernorm(seq_hidden)
                    
                    # Compute Q, K, V
                    q = attn.q_proj(seq_hidden)
                    k = attn.k_proj(seq_hidden)
                    v = attn.v_proj(seq_hidden)
                    
                    # Reshape
                    q = q.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [1, Hq, 1, D]
                    k = k.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, 1, D]
                    v = v.view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, 1, D]
                    
                    # Apply RoPE
                    q, k = self._apply_rope(attn, q, k, seq_position, kv_seq_len)
                    
                    # Get current token's Q, K, V
                    q_tok = q[0, :, 0, :]  # [Hq, D]
                    k_tok = k[0, :, 0, :]  # [Hkv, D]
                    v_tok = v[0, :, 0, :]  # [Hkv, D]
                    
                    # IMPORTANT: append KV first so attention includes self (causal allows self)
                    # This ensures the current token can attend to itself
                    self.paged_attentions[layer_idx].append_kv(
                        seq_id, k_tok, v_tok, seq_pos
                    )
                    
                    # Compute attention using PagedAttention (now includes current token)
                    attn_output = self.paged_attentions[layer_idx].compute_attention(
                        seq_id, q_tok
                    )  # [Hq, D]
                    
                    # Output projection
                    attn_output = attn_output.view(1, 1, -1)
                    attn_output = attn.o_proj(attn_output)
                    
                    # Residual
                    seq_hidden = residual + attn_output
                    
                    # MLP
                    seq_hidden_norm = layer.post_attention_layernorm(seq_hidden)
                    mlp_output = layer.mlp(seq_hidden_norm)
                    seq_hidden = seq_hidden + mlp_output
                
                # Final layer norm and LM head
                seq_hidden = self.model.model.norm(seq_hidden)
                logits = self.model.lm_head(seq_hidden)
                next_token_logits = logits[0, -1, :]
                next_token_logits_list.append(next_token_logits)
            
            # Sample next tokens
            next_token_ids = [
                int(torch.argmax(logits).item())
                for logits in next_token_logits_list
            ]
            
            return next_token_ids
    
    def get_sequence_text(self, seq_id: int) -> Optional[str]:
        """Get generated text for a sequence."""
        if seq_id not in self.scheduler.sequences:
            return None
        
        seq_info = self.scheduler.sequences[seq_id]
        full_tokens = seq_info.prompt_tokens + seq_info.generated_tokens
        return self.tokenizer.decode(full_tokens, skip_special_tokens=True)
    
    def cleanup_finished(self):
        """Clean up finished sequences."""
        finished_ids = self.scheduler.get_finished_sequences()
        for seq_id in finished_ids:
            # Free blocks
            for layer_idx in range(self.num_layers):
                self.paged_attentions[layer_idx].free_sequence(seq_id)
            # Remove from scheduler
            self.scheduler.remove_sequence(seq_id)
        return len(finished_ids)


def main():
    """Main function to demonstrate continuous batching."""
    print("=" * 60)
    print("PagedAttention v3 Inference Demo (Continuous Batching)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = PagedAttentionModelWrapperV3(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        block_size=16,
        device=device,
        use_online_softmax=True,
        max_batch_size=32
    )
    
    # Add multiple requests
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about AI.",
    ]
    
    print(f"\nAdding {len(prompts)} requests to the batch...")
    seq_ids = []
    seq_results = {}  # Store results before cleanup
    
    for i, prompt in enumerate(prompts):
        seq_id = model_wrapper.add_request(prompt, max_new_tokens=50)
        seq_ids.append(seq_id)
        print(f"  Request {i+1}: seq_id={seq_id}, prompt='{prompt[:50]}...'")
    
    # Continuous batching: process steps until all sequences finish
    print(f"\nProcessing continuous batch (max {model_wrapper.scheduler.max_batch_size} sequences)...")
    step = 0
    max_steps = 200
    
    while step < max_steps:
        # Process one step
        processed_seq_ids, next_token_ids = model_wrapper.step()
        
        if not processed_seq_ids:
            break
        
        step += 1
        
        # Store results for finished sequences before cleanup
        finished_ids = model_wrapper.scheduler.get_finished_sequences()
        for seq_id in finished_ids:
            if seq_id not in seq_results:
                text = model_wrapper.get_sequence_text(seq_id)
                seq_results[seq_id] = text
        
        # Print progress every 10 steps
        if step % 10 == 0:
            stats = model_wrapper.scheduler.get_stats()
            print(f"  Step {step}: {stats['active']} active sequences "
                  f"(prefill: {stats['prefill']}, decode: {stats['decode']}, "
                  f"finished: {stats['finished']})")
        
        # Clean up finished sequences
        finished_count = model_wrapper.cleanup_finished()
        if finished_count > 0:
            print(f"  Step {step}: {finished_count} sequence(s) finished")
        
        # Check if all done
        if model_wrapper.scheduler.get_active_count() == 0:
            break
    
    # Print results
    print(f"\n{'=' * 60}")
    print("Generation Results:")
    print(f"{'=' * 60}")
    for i, seq_id in enumerate(seq_ids):
        text = seq_results.get(seq_id, "N/A")
        if text and text != "N/A":
            print(f"\nRequest {i+1} (seq_id={seq_id}):")
            print(f"  {text}")
    
    # Print final stats
    print(f"\n{'=' * 60}")
    print("Final Statistics:")
    print(f"{'=' * 60}")
    stats = model_wrapper.scheduler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    block_stats = model_wrapper.paged_attentions[0].get_stats()
    print(f"\nBlock Manager Stats:")
    for key, value in block_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
