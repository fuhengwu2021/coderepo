"""
TP-aware RadixAttention Model Wrapper.

This integrates Tensor Parallelism with RadixAttention following SGLang's approach:
- TP is applied to QKV projection (column parallel) and output projection (row parallel)
- RadixAttention works with already-sharded Q/K/V tensors
- RadixCache is shared across TP ranks (only rank 0 manages it, or we coordinate)
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List
import sys
import os

# Add parent directory to path for radix imports
radix_path = os.path.join(os.path.dirname(__file__), '..', 'radix')
if radix_path not in sys.path:
    sys.path.insert(0, radix_path)

try:
    from inference_radix_v2 import (
        Router,
        PageManager,
        AttentionExecutor,
    )
    from radix_cache_v2 import EvictionPolicy, MatchResult, TreeNode
except ImportError:
    # Try relative import from parent
    try:
        from radix.inference_radix_v2 import (
            Router,
            PageManager,
            AttentionExecutor,
        )
        from radix.radix_cache_v2 import EvictionPolicy, MatchResult, TreeNode
    except ImportError:
        # Fallback: create minimal stubs (for testing without radix)
        print("Warning: Could not import radix modules. Some features may not work.")
        Router = None
        PageManager = None
        AttentionExecutor = None
        EvictionPolicy = None
        MatchResult = None
        TreeNode = None

try:
    from .parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )
    from .linear import ColumnParallelLinear, RowParallelLinear
except ImportError:
    from parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )
    from linear import ColumnParallelLinear, RowParallelLinear


def apply_tp_to_attention_layer(attention_layer, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
    """
    Apply tensor parallelism to an attention layer.
    
    Replaces QKV projection with ColumnParallelLinear and output projection with RowParallelLinear.
    """
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    
    # Get original QKV projection
    if hasattr(attention_layer, 'q_proj') and hasattr(attention_layer, 'k_proj') and hasattr(attention_layer, 'v_proj'):
        # Separate Q, K, V projections (e.g., Qwen)
        q_proj = attention_layer.q_proj
        k_proj = attention_layer.k_proj
        v_proj = attention_layer.v_proj
        
        # Create TP QKV projections
        # Q: [hidden_size, num_heads * head_dim] -> shard to [hidden_size, (num_heads * head_dim) / tp_size]
        # K: [hidden_size, num_kv_heads * head_dim] -> shard to [hidden_size, (num_kv_heads * head_dim) / tp_size]
        # V: [hidden_size, num_kv_heads * head_dim] -> shard to [hidden_size, (num_kv_heads * head_dim) / tp_size]
        
        q_out_size = num_heads * head_dim
        kv_out_size = num_kv_heads * head_dim
        
        # Store original weights for initialization
        q_weight = q_proj.weight.data.clone()
        k_weight = k_proj.weight.data.clone()
        v_weight = v_proj.weight.data.clone()
        q_bias = q_proj.bias.data.clone() if q_proj.bias is not None else None
        k_bias = k_proj.bias.data.clone() if k_proj.bias is not None else None
        v_bias = v_proj.bias.data.clone() if v_proj.bias is not None else None
        
        # Get device from original layer
        device = next(q_proj.parameters()).device
        
        # Create TP layers
        attention_layer.q_proj = ColumnParallelLinear(
            hidden_size, q_out_size, bias=(q_bias is not None), gather_output=False
        ).to(device)
        attention_layer.k_proj = ColumnParallelLinear(
            hidden_size, kv_out_size, bias=(k_bias is not None), gather_output=False
        ).to(device)
        attention_layer.v_proj = ColumnParallelLinear(
            hidden_size, kv_out_size, bias=(v_bias is not None), gather_output=False
        ).to(device)
        
        # Initialize with sharded weights
        q_out_per_partition = q_out_size // tp_size
        k_out_per_partition = kv_out_size // tp_size
        v_out_per_partition = kv_out_size // tp_size
        
        start_q = tp_rank * q_out_per_partition
        end_q = start_q + q_out_per_partition
        start_k = tp_rank * k_out_per_partition
        end_k = start_k + k_out_per_partition
        start_v = tp_rank * v_out_per_partition
        end_v = start_v + v_out_per_partition
        
        attention_layer.q_proj.weight.data = q_weight[start_q:end_q, :].clone().to(device)
        attention_layer.k_proj.weight.data = k_weight[start_k:end_k, :].clone().to(device)
        attention_layer.v_proj.weight.data = v_weight[start_v:end_v, :].clone().to(device)
        
        if q_bias is not None:
            attention_layer.q_proj.bias.data = q_bias[start_q:end_q].clone().to(device)
        if k_bias is not None:
            attention_layer.k_proj.bias.data = k_bias[start_k:end_k].clone().to(device)
        if v_bias is not None:
            attention_layer.v_proj.bias.data = v_bias[start_v:end_v].clone().to(device)
        
    elif hasattr(attention_layer, 'qkv_proj'):
        # Combined QKV projection (e.g., OPT)
        qkv_proj = attention_layer.qkv_proj
        qkv_weight = qkv_proj.weight.data.clone()
        qkv_bias = qkv_proj.bias.data.clone() if qkv_proj.bias is not None else None
        
        total_qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
        qkv_out_per_partition = total_qkv_size // tp_size
        
        # Get device from original layer
        device = next(qkv_proj.parameters()).device
        
        attention_layer.qkv_proj = ColumnParallelLinear(
            hidden_size, total_qkv_size, bias=(qkv_bias is not None), gather_output=False
        ).to(device)
        
        start = tp_rank * qkv_out_per_partition
        end = start + qkv_out_per_partition
        attention_layer.qkv_proj.weight.data = qkv_weight[start:end, :].clone().to(device)
        if qkv_bias is not None:
            attention_layer.qkv_proj.bias.data = qkv_bias[start:end].clone().to(device)
    
    # Apply TP to output projection
    if hasattr(attention_layer, 'o_proj'):
        o_proj = attention_layer.o_proj
        o_weight = o_proj.weight.data.clone()
        o_bias = o_proj.bias.data.clone() if o_proj.bias is not None else None
        
        # Get device from original layer
        device = next(o_proj.parameters()).device
        
        attention_layer.o_proj = RowParallelLinear(
            hidden_size, hidden_size, bias=(o_bias is not None), 
            input_is_parallel=True, reduce_results=True
        ).to(device)
        
        # Shard input dimension (rows)
        o_in_per_partition = hidden_size // tp_size
        start = tp_rank * o_in_per_partition
        end = start + o_in_per_partition
        
        # Get device from original layer
        device = next(o_proj.parameters()).device
        
        attention_layer.o_proj.weight.data = o_weight[:, start:end].clone().to(device)
        if o_bias is not None:
            attention_layer.o_proj.bias.data = o_bias.clone().to(device)  # Bias not sharded


class TPRadixAttentionModelWrapper:
    """
    TP-aware RadixAttention Model Wrapper.
    
    This extends RadixAttentionModelWrapperV2 with Tensor Parallelism support.
    Following SGLang's approach:
    - TP is applied to attention layers (QKV and output projection)
    - RadixCache is managed on rank 0 (or coordinated across ranks)
    - All TP ranks participate in inference
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        page_size: int = 16,
    ):
        """
        Initialize the TP-aware RadixAttention model wrapper.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            page_size: Page size for cache
        """
        self.device = device
        self.page_size = page_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.is_main_rank = (self.tp_rank == 0)
        
        if self.is_main_rank:
            print(f"Loading model {model_name} on TP rank {self.tp_rank}...")
        
        # Load model and tokenizer (all ranks load, but we'll shard weights)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set device for this TP rank
        if device == "cuda" and torch.cuda.is_available():
            device_id = self.tp_rank % torch.cuda.device_count()
            self.device = f"cuda:{device_id}"
        else:
            self.device = device
        
        # Load model without device_map to have more control
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        # Move to device after loading
        if self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model config
        config = self.model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        
        # Adjust num_heads per TP rank
        assert self.num_heads % self.tp_size == 0, f"num_heads ({self.num_heads}) must be divisible by tp_size ({self.tp_size})"
        self.num_heads_per_rank = self.num_heads // self.tp_size
        
        if self.is_main_rank:
            print(f"Model config: {self.num_heads} Q heads (/{self.tp_size} = {self.num_heads_per_rank} per rank), "
                  f"{self.num_kv_heads} KV heads, {self.head_dim} head_dim, {self.num_layers} layers")
            print(f"Using TP RadixAttention with page_size={page_size}")
        
        # Apply TP to attention layers
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            attention = layer.self_attn
            apply_tp_to_attention_layer(
                attention, self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim
            )
        
        # Initialize RadixCache components (only on main rank, or coordinate)
        if self.is_main_rank and Router is not None and PageManager is not None and EvictionPolicy is not None:
            self.router = Router(
                self.num_layers,
                page_size=page_size,
                device=self.device,
                max_pages=10000,
                eviction_policy=EvictionPolicy.LRU
            )
            self.page_manager = PageManager(
                self.num_layers, self.num_heads, self.head_dim, 
                page_size=page_size, device=self.device
            )
        else:
            # Other ranks don't manage cache (simplified approach)
            # In a full implementation, we'd coordinate cache across ranks
            self.router = None
            self.page_manager = None
        
        # All ranks have executor (but it uses TP-sharded model)
        if AttentionExecutor is not None:
            self.executor = AttentionExecutor(
                self.model, self.num_heads_per_rank, self.num_kv_heads, 
                self.head_dim, self.num_layers, device=self.device
            )
        else:
            self.executor = None
        
        # Track sequences
        self.sequences: Dict[int, Dict] = {}
        self.next_seq_id = 0
        
        # Per-sequence KV cache for decode
        self.kv_caches: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    
    def prefill(self, prompt: str, seq_id: Optional[int] = None) -> int:
        """
        Process the prompt with prefix cache reuse and TP.
        
        Only main rank manages RadixCache, but all ranks participate in computation.
        """
        if seq_id is None:
            seq_id = self.next_seq_id
            self.next_seq_id += 1
        
        # Tokenize (all ranks do this)
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
        
        if self.is_main_rank:
            print(f"\n[Prefill] Sequence {seq_id}: Processing {len(prompt_tokens)} prompt tokens with TP={self.tp_size}")
        
        # Match prefix in radix cache (only main rank manages cache)
        match_results = []
        if self.is_main_rank and self.router is not None:
            for layer_idx in range(self.num_layers):
                match_result = self.router.match_prefix(layer_idx, prompt_tokens)
                match_results.append(match_result)
        else:
            # Other ranks: create dummy match results (or broadcast from main rank)
            for layer_idx in range(self.num_layers):
                if MatchResult is not None and TreeNode is not None:
                    # Create proper MatchResult with required parameters
                    dummy_node = TreeNode()
                    match_results.append(MatchResult(
                        page_ids=torch.tensor([], dtype=torch.long, device=self.device),
                        last_device_node=dummy_node
                    ))
                else:
                    # Fallback if imports fail
                    class DummyTreeNode:
                        def __init__(self):
                            self.children = {}
                            self.value = None
                    class DummyMatchResult:
                        def __init__(self):
                            self.page_ids = torch.tensor([], dtype=torch.long, device=self.device)
                            self.device_indices = self.page_ids
                            self.last_device_node = DummyTreeNode()
                    match_results.append(DummyMatchResult())
        
        # Broadcast match results to all ranks (simplified: just use empty for now)
        # In full implementation, we'd broadcast from main rank
        
        # Forward pass with TP
        with torch.no_grad():
            outputs = self.model(input_ids=tokens, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Store KV cache (main rank manages pages, all ranks store per-sequence cache)
        self.kv_caches[seq_id] = {}
        for layer_idx in range(self.num_layers):
            k, v = past_key_values[layer_idx]
            # k, v are already sharded per TP rank
            k_cache = k[0]  # [num_kv_heads_per_rank, seq_len, head_dim]
            v_cache = v[0]
            
            self.kv_caches[seq_id][layer_idx] = {
                'k': k_cache,
                'v': v_cache,
            }
        
        # Main rank updates RadixCache
        if self.is_main_rank and self.router is not None and self.page_manager is not None:
            for layer_idx in range(self.num_layers):
                # For simplicity, we'll insert the full prefix
                # In full implementation, we'd handle page allocation properly
                pass
        
        return seq_id
    
    def decode_step(self, seq_id: int) -> int:
        """
        Generate one token with TP.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Generated token ID
        """
        # Get last token
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        last_token_id = self.sequences[seq_id]["generated_tokens"][-1] if self.sequences[seq_id]["generated_tokens"] else None
        if last_token_id is None:
            # Use last prompt token
            prompt_tokens = self.sequences[seq_id]["prompt_tokens"]
            last_token_id = prompt_tokens[-1]
        
        input_ids = torch.tensor([[last_token_id]], device=self.device)
        
        # Forward pass with cached KV
        # Convert our cache format to transformers DynamicCache format
        with torch.no_grad():
            # Try to use DynamicCache
            cache = None
            try:
                from transformers.cache_utils import DynamicCache
                
                # Create DynamicCache directly
                cache = DynamicCache()
                for layer_idx in range(self.num_layers):
                    k_cache = self.kv_caches[seq_id][layer_idx]['k']  # [num_kv_heads, seq_len, head_dim]
                    v_cache = self.kv_caches[seq_id][layer_idx]['v']
                    
                    # DynamicCache expects [batch, num_heads, seq_len, head_dim]
                    # Add batch dimension
                    k_batch = k_cache.unsqueeze(0)  # [1, num_kv_heads, seq_len, head_dim]
                    v_batch = v_cache.unsqueeze(0)
                    
                    # Update cache using the update method
                    cache.update(k_batch, v_batch, layer_idx)
                
            except (ImportError, AttributeError, TypeError) as e:
                # If DynamicCache doesn't work, we'll try legacy format
                if self.is_main_rank:
                    print(f"Warning: Could not use DynamicCache: {e}, trying legacy format")
                cache = None
            
            # Use the cache (DynamicCache or None)
            if cache is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    use_cache=True
                )
                
                # Update our cache from DynamicCache
                past_key_values = outputs.past_key_values
                
                # Try to convert to legacy format first (simplest approach)
                try:
                    if hasattr(past_key_values, 'to_legacy_cache'):
                        legacy_cache = past_key_values.to_legacy_cache()
                        for layer_idx in range(self.num_layers):
                            k, v = legacy_cache[layer_idx]
                            # Remove batch dimension if present: [1, num_heads, seq_len, head_dim] -> [num_heads, seq_len, head_dim]
                            if k.dim() == 4:
                                k = k[0]
                            if v.dim() == 4:
                                v = v[0]
                            self.kv_caches[seq_id][layer_idx]['k'] = k
                            self.kv_caches[seq_id][layer_idx]['v'] = v
                    elif isinstance(past_key_values, list):
                        # Already in legacy format
                        for layer_idx in range(self.num_layers):
                            k, v = past_key_values[layer_idx]
                            if k.dim() == 4:
                                k = k[0]
                            if v.dim() == 4:
                                v = v[0]
                            self.kv_caches[seq_id][layer_idx]['k'] = k
                            self.kv_caches[seq_id][layer_idx]['v'] = v
                    else:
                        # Try accessing as DynamicCache with different attribute names
                        # Some versions use different names
                        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
                            for layer_idx in range(self.num_layers):
                                k = past_key_values.key_cache[layer_idx]
                                v = past_key_values.value_cache[layer_idx]
                                self.kv_caches[seq_id][layer_idx]['k'] = k[0] if k.dim() == 4 else k
                                self.kv_caches[seq_id][layer_idx]['v'] = v[0] if v.dim() == 4 else v
                        else:
                            # Last resort: access by index
                            for layer_idx in range(self.num_layers):
                                item = past_key_values[layer_idx]
                                if isinstance(item, tuple):
                                    k, v = item
                                else:
                                    raise RuntimeError(f"Unexpected cache item type: {type(item)}")
                                
                                if k.dim() == 4:
                                    k = k[0]
                                if v.dim() == 4:
                                    v = v[0]
                                self.kv_caches[seq_id][layer_idx]['k'] = k
                                self.kv_caches[seq_id][layer_idx]['v'] = v
                except Exception as e:
                    if self.is_main_rank:
                        print(f"Error extracting cache: {e}")
                        print(f"Cache type: {type(past_key_values)}")
                        print(f"Cache attributes: {dir(past_key_values)}")
                    raise
            else:
                # Fallback: Create DynamicCache using from_legacy_cache if available
                # Otherwise, we need to manually construct it
                try:
                    from transformers.cache_utils import DynamicCache
                    
                    # Try from_legacy_cache method
                    past_key_values_list = []
                    for layer_idx in range(self.num_layers):
                        k_cache = self.kv_caches[seq_id][layer_idx]['k']
                        v_cache = self.kv_caches[seq_id][layer_idx]['v']
                        k_batch = k_cache.unsqueeze(0)
                        v_batch = v_cache.unsqueeze(0)
                        past_key_values_list.append((k_batch, v_batch))
                    
                    # Try to use from_legacy_cache if it exists
                    if hasattr(DynamicCache, 'from_legacy_cache'):
                        cache = DynamicCache.from_legacy_cache(past_key_values_list)
                    else:
                        # Manually create cache
                        cache = DynamicCache()
                        for layer_idx, (k, v) in enumerate(past_key_values_list):
                            cache.update(k, v, layer_idx)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=cache,
                        use_cache=True
                    )
                    
                    # Update KV cache - use same extraction logic
                    past_key_values = outputs.past_key_values
                    
                    try:
                        if hasattr(past_key_values, 'to_legacy_cache'):
                            legacy_cache = past_key_values.to_legacy_cache()
                            for layer_idx in range(self.num_layers):
                                k, v = legacy_cache[layer_idx]
                                if k.dim() == 4:
                                    k = k[0]
                                if v.dim() == 4:
                                    v = v[0]
                                self.kv_caches[seq_id][layer_idx]['k'] = k
                                self.kv_caches[seq_id][layer_idx]['v'] = v
                        elif isinstance(past_key_values, list):
                            for layer_idx in range(self.num_layers):
                                k, v = past_key_values[layer_idx]
                                if k.dim() == 4:
                                    k = k[0]
                                if v.dim() == 4:
                                    v = v[0]
                                self.kv_caches[seq_id][layer_idx]['k'] = k
                                self.kv_caches[seq_id][layer_idx]['v'] = v
                        else:
                            # Try accessing by index
                            for layer_idx in range(self.num_layers):
                                item = past_key_values[layer_idx]
                                if isinstance(item, tuple):
                                    k, v = item
                                else:
                                    raise RuntimeError(f"Unexpected cache item type: {type(item)}")
                                
                                if k.dim() == 4:
                                    k = k[0]
                                if v.dim() == 4:
                                    v = v[0]
                                self.kv_caches[seq_id][layer_idx]['k'] = k
                                self.kv_caches[seq_id][layer_idx]['v'] = v
                    except Exception as e:
                        if self.is_main_rank:
                            print(f"Error extracting cache in fallback: {e}")
                            print(f"Cache type: {type(past_key_values)}")
                        raise
                except Exception as e:
                    if self.is_main_rank:
                        print(f"Error in decode_step cache handling: {e}")
                    raise
            
            # Get logits (all-reduce happens in output projection, so logits are same on all ranks)
            logits = outputs.logits[0, -1, :]
        
        # Sample token (all ranks do this, but should be deterministic)
        token_id = torch.argmax(logits, dim=-1).item()
        
        # Update sequence
        self.sequences[seq_id]["generated_tokens"].append(token_id)
        self.sequences[seq_id]["total_tokens"] += 1
        
        return token_id
    
    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate text with TP and RadixAttention.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        seq_id = self.prefill(prompt)
        
        generated_tokens = []
        for _ in range(max_new_tokens):
            token_id = self.decode_step(seq_id)
            generated_tokens.append(token_id)
            
            if self.is_main_rank:
                token = self.tokenizer.decode([token_id], skip_special_tokens=True)
                print(f"[Generate] Token: {token} (id={token_id})")
        
        if self.is_main_rank:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text
        else:
            # Other ranks: could broadcast from main rank, or decode locally
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return generated_text
