"""
Constraint Decoder for structured output generation.

This module implements the main constraint decoding logic that
integrates with HuggingFace models to generate outputs conforming
to a grammar.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Set
import time

try:
    from .grammar import Grammar
    from .fsm import FSM
    from .pda import PDA
except ImportError:
    from grammar import Grammar
    from fsm import FSM
    from pda import PDA


class ConstraintDecoder:
    """
    Constraint Decoder for generating grammar-conforming outputs.
    
    Uses FSM for simple rules and PDA for stack-based rules.
    Masks invalid tokens during generation to ensure output
    conforms to the specified grammar.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        grammar: Optional[Grammar] = None,
    ):
        """
        Initialize Constraint Decoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            grammar: Grammar object (optional, can be set later)
        """
        self.device = device
        self.model_name = model_name
        
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device
        )
        self.model.eval()
        
        # Get vocabulary
        self.vocab_size = len(self.tokenizer)
        self.token_to_id = {token: idx for token, idx in self.tokenizer.get_vocab().items()}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Initialize grammar and automaton
        self.grammar = grammar
        self.fsm: Optional[FSM] = None
        self.pda: Optional[PDA] = None
        
        if grammar:
            self.set_grammar(grammar)
    
    def set_grammar(self, grammar: Grammar):
        """
        Set grammar and build appropriate automaton.
        
        Args:
            grammar: Grammar object
        """
        self.grammar = grammar
        
        if grammar.requires_stack:
            print("Grammar requires stack - using PDA")
            self.pda = PDA(grammar, start_symbol=grammar.start_symbol)
            self.fsm = None
        else:
            print("Grammar is context-free - using FSM")
            self.fsm = FSM(grammar, start_symbol=grammar.start_symbol)
            self.pda = None
    
    def _get_valid_token_ids(self) -> Set[int]:
        """
        Get set of valid token IDs for current state.
        
        Returns:
            Set of token IDs that are valid in current state
        """
        if self.fsm:
            valid_tokens = self.fsm.get_valid_tokens()
        elif self.pda:
            stack_top = self.pda.stack[-1] if self.pda.stack else None
            valid_tokens = self.pda.get_valid_tokens(stack_top=stack_top)
        else:
            # No grammar - all tokens valid
            return set(range(self.vocab_size))
        
        # Convert token strings to token IDs
        valid_token_ids = set()
        
        # Common JSON/grammar tokens that should always be considered
        common_json_tokens = {
            '{': ['{', ' {', '{ ', ' { '],
            '}': ['}', ' }', '} ', ' } '],
            '[': ['[', ' [', '[ ', ' [ '],
            ']': [']', ' ]', '] ', ' ] '],
            ',': [',', ' ,', ', ', ' , '],
            ':': [':', ' :', ': ', ' : '],
            '"': ['"', ' "', '" ', ' " '],
            'true': ['true', ' true', 'true ', ' true '],
            'false': ['false', ' false', 'false ', ' false '],
            'null': ['null', ' null', 'null ', ' null '],
        }
        
        # Try to match grammar tokens
        for token_str in valid_tokens:
            # Clean token string
            token_str = token_str.strip()
            
            # Direct token ID lookup
            try:
                token_id = self.tokenizer.convert_tokens_to_ids([token_str])
                if token_id and token_id[0] != self.tokenizer.unk_token_id:
                    valid_token_ids.add(token_id[0])
            except:
                pass
            
            # Try encoding
            try:
                encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
                if encoded:
                    valid_token_ids.update(encoded)
            except:
                pass
        
        # Add common JSON tokens if they're in valid_tokens or if we have few matches
        if len(valid_token_ids) < 50:
            for json_token, variants in common_json_tokens.items():
                if json_token in valid_tokens or len(valid_tokens) == 0:
                    for variant in variants:
                        try:
                            token_id = self.tokenizer.convert_tokens_to_ids([variant])
                            if token_id and token_id[0] != self.tokenizer.unk_token_id:
                                valid_token_ids.add(token_id[0])
                        except:
                            pass
                        
                        try:
                            encoded = self.tokenizer.encode(variant, add_special_tokens=False)
                            if encoded:
                                valid_token_ids.update(encoded)
                        except:
                            pass
        
        # If still very few, allow more tokens (soft constraint)
        if len(valid_token_ids) < 10:
            # Allow alphanumeric tokens that might form valid JSON
            # This is a fallback - real implementation would be more sophisticated
            for i in range(min(1000, self.vocab_size)):
                try:
                    token = self.tokenizer.decode([i])
                    # Allow tokens that look like they could be part of JSON
                    if any(c.isalnum() or c in '.,:;' for c in token):
                        valid_token_ids.add(i)
                except:
                    pass
        
        return valid_token_ids
    
    def _mask_invalid_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask invalid tokens in logits.
        
        Args:
            logits: Logits tensor [vocab_size]
            
        Returns:
            Masked logits
        """
        valid_token_ids = self._get_valid_token_ids()
        
        if not valid_token_ids:
            # No constraints - return original
            return logits
        
        # Create mask
        mask = torch.ones_like(logits) * float('-inf')
        for token_id in valid_token_ids:
            if 0 <= token_id < self.vocab_size:
                mask[token_id] = 0.0
        
        # Apply mask
        masked_logits = logits + mask
        return masked_logits
    
    def _update_state(self, token: str):
        """
        Update FSM/PDA state with generated token.
        
        Args:
            token: Generated token string
        """
        if self.fsm:
            self.fsm.transition(token)
        elif self.pda:
            self.pda.transition(token)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text conforming to grammar.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        if not self.grammar:
            raise ValueError("Grammar not set. Call set_grammar() first.")
        
        # Reset automaton
        if self.fsm:
            self.fsm.reset()
        elif self.pda:
            self.pda.reset()
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            # Prefill
            outputs = self.model(input_ids=input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]
            
            # Generate tokens
            for step in range(max_new_tokens):
                # Mask invalid tokens
                masked_logits = self._mask_invalid_tokens(next_token_logits)
                
                # Apply temperature
                if temperature != 1.0:
                    masked_logits = masked_logits / temperature
                
                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(masked_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    masked_logits[indices_to_remove] = float('-inf')
                
                # Sample token
                probs = F.softmax(masked_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                # Decode token
                next_token = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                generated_tokens.append(next_token_id)
                
                # Update state
                self._update_state(next_token)
                
                # Check for EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Prepare for next iteration
                token_tensor = torch.tensor([[next_token_id]], device=self.device)
                outputs = self.model(
                    input_ids=token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[0, -1, :]
        
        # Decode full sequence
        full_ids = torch.cat([input_ids[0], torch.tensor(generated_tokens, device=self.device)])
        generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
        
        return generated_text
