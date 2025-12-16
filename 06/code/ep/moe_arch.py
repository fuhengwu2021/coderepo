"""pip install einops flash_attn accelerate
use conda env codellm
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.cache_utils import DynamicCache

# Monkey patch: Add compatibility methods to DynamicCache for outdated custom code
# This fixes AttributeErrors when using trust_remote_code=True with Phi-tiny-MoE-instruct
if not hasattr(DynamicCache, 'seen_tokens'):
    @property
    def seen_tokens(self):
        """Compatibility property: maps to get_seq_length() for outdated custom code."""
        return self.get_seq_length()
    DynamicCache.seen_tokens = seen_tokens

if not hasattr(DynamicCache, 'get_usable_length'):
    def get_usable_length(self, seq_length, layer_idx=None):
        """Compatibility method: returns the usable cache length (same as get_seq_length())."""
        return self.get_seq_length()
    DynamicCache.get_usable_length = get_usable_length

torch.random.manual_seed(0) 

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-tiny-MoE-instruct",  
    device_map="cuda",  
    dtype="auto",  
    trust_remote_code=True,  # Required for Phi-tiny-MoE-instruct (monkey-patched DynamicCache for compatibility)
)
from hiq.vis import print_model
print_model(model)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-tiny-MoE-instruct") 

messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])


