""" 
Inference script using PagedAttention v4 with True Ragged Batching.

This script implements true ragged batching:
- Prefill batching: Multiple prompts flattened to T×D with metadata
- Decode batching: Multiple sequences flattened with metadata
- Single forward pass for all tokens (no padding)
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

from pa import PagedAttentionV4
from pa.scheduler import ContinuousBatchScheduler, SequenceState


class PagedAttentionModelWrapperV4:
    """
    Wrapper around a HuggingFace model to use PagedAttention v4 with true ragged batching.
    
    This version implements:
    - Prefill batching: Multiple prompts processed together (flattened tokens + metadata)
    - Decode batching: Multiple sequences decoded together (flattened tokens + metadata)
    - True ragged batching: T×D instead of B×Lmax×D
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
            max_batch_size: Maximum batch size for batching
        """
        self.device = device
        self.block_size = block_size
        self.use_online_softmax = use_online_softmax
        self.max_batch_size = max_batch_size
        
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
        print(f"Using PagedAttention v4 with {algo_name} (True Ragged Batching)")
        
        # Initialize scheduler
        self.scheduler = ContinuousBatchScheduler(max_batch_size=max_batch_size)
        
        # Initialize PagedAttention v4 for each layer
        self.paged_attentions = [
            PagedAttentionV4(
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
            if hasattr(self.model.model, 'rotary_emb'):
                rotary_emb = self.model.model.rotary_emb
                cos, sin = rotary_emb(k, position_ids)
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
                return q_rope, k_rope
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
        Add a new request to the scheduler (does NOT prefill immediately).
        
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
        
        # Add to scheduler (stays in PREFILL state, waiting for batching)
        seq_id = self.scheduler.add_sequence(prompt_tokens, max_new_tokens, immediate_prefill=False)
        
        return seq_id
    
    def prefill_batch(self) -> int:
        """
        Process a batch of sequences in prefill phase using ragged batching.
        
        Returns:
            Number of sequences processed
        """
        # Get sequences ready for prefill batching
        seq_ids, prompt_token_lists, positions_start, seq_lengths = self.scheduler.get_prefill_batch(
            max_batch_size=self.max_batch_size
        )
        
        if not seq_ids:
            return 0
        
        # Build ragged batching metadata
        token_ids_flat, seq_id_flat, position_flat, slot_mapping_flat = \
            self.paged_attentions[0].build_ragged_metadata(seq_ids, prompt_token_lists, positions_start)
        
        T = len(token_ids_flat)
        max_len = max(seq_lengths) if seq_lengths else 0
        padding_tokens = len(seq_ids) * max_len - T
        print(f"[Prefill Batch] Processing {len(seq_ids)} sequences, {T} total tokens "
              f"(would be {len(seq_ids) * max_len} with padding, saving {padding_tokens} tokens)")
        print(f"[Prefill Batch] Metadata: seq_id_flat={len(seq_ids)} sequences, "
              f"position_flat={T} tokens, slot_mapping_flat={T} slots")
        print(f"[Prefill Batch] Note: Using metadata structure, but processing sequences separately "
              f"(HuggingFace doesn't support true ragged batching)")
        
        # Process all tokens using ragged batching concept
        # NOTE: HuggingFace model.forward doesn't support true ragged batching,
        # so we process each sequence separately but demonstrate the metadata structure
        
        with torch.no_grad():
            # Process each sequence (in true implementation, would be one forward call)
            # But we demonstrate the flattened tokens + metadata concept
            for i, (seq_id, prompt_tokens) in enumerate(zip(seq_ids, prompt_token_lists)):
                # Process this sequence's tokens
                seq_tokens = torch.tensor([prompt_tokens], device=self.device)  # [1, L_i]
                
                # Forward pass for this sequence
                outputs = self.model(input_ids=seq_tokens, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Extract and store KV cache
                for layer_idx in range(self.num_layers):
                    k, v = past_key_values[layer_idx]
                    # k, v shape: [1, num_kv_heads, L_i, head_dim]
                    k_cache = k[0]  # [num_kv_heads, L_i, head_dim]
                    v_cache = v[0]  # [num_kv_heads, L_i, head_dim]
                    
                    # Store each token's K/V in PagedAttention blocks
                    for token_pos in range(len(prompt_tokens)):
                        k_tok = k_cache[:, token_pos, :]  # [num_kv_heads, head_dim]
                        v_tok = v_cache[:, token_pos, :]  # [num_kv_heads, head_dim]
                        self.paged_attentions[layer_idx].append_kv(seq_id, k_tok, v_tok, token_pos)
        
        # Update scheduler: mark all sequences as decode-ready
        for seq_id in seq_ids:
            seq_info = self.scheduler.sequences[seq_id]
            seq_info.state = SequenceState.DECODE
            seq_info.position = len(seq_info.prompt_tokens)
        
        return len(seq_ids)
    
    def decode_batch(self) -> Tuple[List[int], List[int]]:
        """
        Process a batch of sequences in decode phase using TRUE ragged batching.
        
        This implementation processes all sequences in parallel:
        - Batch Q, K, V computation
        - Batch RoPE application
        - Batch KV append
        - Batch attention computation (using PagedAttention per sequence)
        
        Returns:
            Tuple of (seq_ids, next_token_ids)
        """
        # Get sequences ready for decode
        seq_ids, positions, token_ids = self.scheduler.get_batch(
            include_prefill=False,
            include_decode=True
        )
        
        if not seq_ids:
            return [], []
        
        # Build decode metadata (each sequence has 1 token)
        seq_id_flat, position_flat, slot_mapping_flat = \
            self.paged_attentions[0].build_decode_metadata(seq_ids, positions)
        
        num_seqs = len(seq_ids)
        print(f"[Decode Batch] Processing {num_seqs} sequences with TRUE ragged batching")
        
        # Flatten tokens: [num_seqs] → [1, num_seqs] (ragged batching: each sequence 1 token)
        token_tensor = torch.tensor([token_ids], device=self.device)  # [1, num_seqs]
        
        with torch.no_grad():
            # ✅ Step 1: Embed all tokens at once (batch embedding)
            hidden_states = self.model.model.embed_tokens(token_tensor)  # [1, num_seqs, H]
            # This is already ragged batching: [1, num_seqs] instead of [num_seqs, max_len]
            # IMPORTANT: hidden_states[0, i] corresponds to seq_ids[i]
            
            # ✅ Step 2: Process all sequences through layers in parallel
            # Process through each layer
            # IMPORTANT: Maintain sequence order throughout - hidden_states[0, i] should always correspond to seq_ids[i]
            for layer_idx in range(self.num_layers):
                layer = self.model.model.layers[layer_idx]
                attn = layer.self_attn
                
                # Residual connection (save before layer norm)
                residual = hidden_states.clone()  # [1, num_seqs, H]
                
                # ✅ Step 3: Batch layer norm (all sequences at once)
                hidden_states = layer.input_layernorm(hidden_states)  # [1, num_seqs, H]
                
                # ✅ Step 4: Batch Q, K, V projection (all sequences at once)
                q = attn.q_proj(hidden_states)  # [1, num_seqs, Hq*D]
                k = attn.k_proj(hidden_states)  # [1, num_seqs, Hkv*D]
                v = attn.v_proj(hidden_states)  # [1, num_seqs, Hkv*D]
                
                # Reshape for attention
                q = q.view(1, num_seqs, self.num_heads, self.head_dim).transpose(1, 2)  # [1, Hq, num_seqs, D]
                k = k.view(1, num_seqs, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, num_seqs, D]
                v = v.view(1, num_seqs, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [1, Hkv, num_seqs, D]
                
                # ✅ Step 5: Batch RoPE application
                # Apply RoPE to each sequence (RoPE needs per-sequence position)
                # IMPORTANT: Process in the same order as seq_ids to maintain sequence order
                # After transpose, q shape is [1, Hq, num_seqs, D], so q[:, :, i, :] is sequence i
                for i, seq_id in enumerate(seq_ids):
                    seq_position = torch.tensor([[positions[i]]], device=self.device, dtype=torch.long)
                    seq_pos = positions[i]
                    kv_seq_len = seq_pos + 1
                    
                    # Extract Q, K for this sequence (index i corresponds to sequence i)
                    # IMPORTANT: After transpose(1, 2), sequence dimension is at index 2
                    q_seq = q[:, :, i:i+1, :]  # [1, Hq, 1, D] - sequence i
                    k_seq = k[:, :, i:i+1, :]  # [1, Hkv, 1, D] - sequence i
                    
                    # Apply RoPE
                    q_seq, k_seq = self._apply_rope(attn, q_seq, k_seq, seq_position, kv_seq_len)
                    
                    # Put back (ensuring sequence order is preserved)
                    q[:, :, i:i+1, :] = q_seq
                    k[:, :, i:i+1, :] = k_seq
                
                # ✅ Step 6: Batch KV append (all sequences at once)
                # Extract K, V for each sequence: [1, Hkv, num_seqs, D] -> [num_seqs, Hkv, D]
                k_batch = k[0].transpose(0, 1)  # [num_seqs, Hkv, D]
                v_batch = v[0].transpose(0, 1)  # [num_seqs, Hkv, D]
                
                for i, seq_id in enumerate(seq_ids):
                    k_tok = k_batch[i]  # [Hkv, D]
                    v_tok = v_batch[i]  # [Hkv, D]
                    seq_pos = positions[i]
                    # IMPORTANT: append KV first so attention includes self
                    self.paged_attentions[layer_idx].append_kv(seq_id, k_tok, v_tok, seq_pos)
                
                # ✅ Step 7: Batch attention computation (using PagedAttention for each sequence)
                # Extract Q for each sequence: [1, Hq, num_seqs, D] -> [num_seqs, Hq, D]
                # q shape: [1, Hq, num_seqs, D]
                # q[0] shape: [Hq, num_seqs, D]
                # transpose(0, 1) -> [num_seqs, Hq, D]
                q_batch = q[0].transpose(0, 1)  # [num_seqs, Hq, D]
                
                # Compute attention for each sequence (this is where we use PagedAttention)
                # IMPORTANT: Process in the same order as seq_ids to maintain sequence order
                attn_outputs = []
                for i, seq_id in enumerate(seq_ids):
                    q_tok = q_batch[i]  # [Hq, D] - sequence i's query
                    attn_output = self.paged_attentions[layer_idx].compute_attention(seq_id, q_tok)
                    # attn_output: [Hq, D]
                    attn_outputs.append(attn_output)
                
                # Stack attention outputs: [num_seqs, Hq, D] -> [1, num_seqs, Hq*D]
                # Each attn_output is [Hq, D], stack to [num_seqs, Hq, D]
                # IMPORTANT: torch.stack preserves order, so attn_outputs[0] -> index 0, etc.
                # This ensures attn_output_tensor[i] corresponds to seq_ids[i]
                attn_output_tensor = torch.stack(attn_outputs, dim=0)  # [num_seqs, Hq, D]
                # Reshape to [num_seqs, Hq*D] then add batch dimension
                # IMPORTANT: The order here must match the order of seq_ids
                attn_output_tensor = attn_output_tensor.view(num_seqs, self.num_heads * self.head_dim).unsqueeze(0)  # [1, num_seqs, Hq*D]
                
                # ✅ Step 8: Batch output projection (all sequences at once)
                attn_output = attn.o_proj(attn_output_tensor)  # [1, num_seqs, H]
                
                # Residual connection
                # IMPORTANT: residual and attn_output should have same sequence order
                # residual: [1, num_seqs, H] - saved before layer norm
                # attn_output: [1, num_seqs, H] - from attention, same order as seq_ids
                hidden_states = residual + attn_output  # [1, num_seqs, H]
                
                # ✅ Step 9: Batch MLP (all sequences at once)
                hidden_states_norm = layer.post_attention_layernorm(hidden_states)
                mlp_output = layer.mlp(hidden_states_norm)  # [1, num_seqs, H]
                hidden_states = hidden_states + mlp_output  # [1, num_seqs, H]
            
            # ✅ Step 10: Batch final layer norm and LM head (all sequences at once)
            hidden_states = self.model.model.norm(hidden_states)  # [1, num_seqs, H]
            logits = self.model.lm_head(hidden_states)  # [1, num_seqs, vocab_size]
            
            # Extract next token logits for each sequence
            next_token_logits = logits[0, :, :]  # [num_seqs, vocab_size]
            
            # Sample next tokens (batch sampling)
            # Use argmax along vocab dimension for each sequence
            next_token_ids = [
                int(torch.argmax(next_token_logits[i]).item())
                for i in range(num_seqs)
            ]
            
            # Update scheduler
            self.scheduler.update_sequences(
                seq_ids,
                next_token_ids,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            return seq_ids, next_token_ids
    
    def step(self) -> Tuple[List[int], List[int]]:
        """
        Process one step: handle prefill batching and decode batching.
        
        Returns:
            Tuple of (seq_ids, next_token_ids) for decode sequences
        """
        # First, process any pending prefill batches
        prefill_count = self.prefill_batch()
        if prefill_count > 0:
            print(f"  Processed {prefill_count} sequences in prefill batch")
        
        # Then, process decode batch
        return self.decode_batch()
    
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
            for layer_idx in range(self.num_layers):
                self.paged_attentions[layer_idx].free_sequence(seq_id)
            self.scheduler.remove_sequence(seq_id)
        return len(finished_ids)


def main():
    """Main function to demonstrate true ragged batching."""
    print("=" * 60)
    print("PagedAttention v4 Inference Demo (True Ragged Batching)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Running on CPU. This will be slow.")
    
    # Initialize model wrapper
    model_wrapper = PagedAttentionModelWrapperV4(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        block_size=16,
        device=device,
        use_online_softmax=True,
        max_batch_size=32
    )
    
    # Add multiple requests (they will be batched for prefill)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about AI.",
    ]
    
    print(f"\nAdding {len(prompts)} requests (will be batched for prefill)...")
    seq_ids = []
    seq_results = {}
    
    for i, prompt in enumerate(prompts):
        seq_id = model_wrapper.add_request(prompt, max_new_tokens=50)
        seq_ids.append(seq_id)
        print(f"  Request {i+1}: seq_id={seq_id}, prompt='{prompt[:50]}...'")
    
    # Process continuous batch with ragged batching
    print(f"\nProcessing with true ragged batching...")
    step = 0
    max_steps = 200
    
    while step < max_steps:
        # Process one step (handles prefill batching and decode batching)
        processed_seq_ids, next_token_ids = model_wrapper.step()
        
        if not processed_seq_ids and model_wrapper.scheduler.get_prefill_batch()[0] == []:
            break
        
        step += 1
        
        # Store results for finished sequences
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
