# Constraint Decoding (X-Grammar) Implementation Summary

## Overview

This implementation provides a simplified Python version of SGLang's X-Grammar constraint decoding system. It enables LLMs to generate outputs that conform to Context-Free Grammars (CFG) using Finite State Machines (FSMs) and Pushdown Automata (PDAs).

## Architecture

### Core Components

1. **Grammar Parser** (`grammar.py`)
   - Parses CFG rules from text
   - Identifies terminals and non-terminals
   - Detects if grammar requires stack (PDA) or can use FSM

2. **Finite State Machine** (`fsm.py`)
   - For simple context-free rules
   - Precompiles valid tokens for each state
   - Efficient token validation

3. **Pushdown Automaton** (`pda.py`)
   - For stack-based rules (nested structures)
   - Handles context-dependent validation
   - Stack management for balanced structures

4. **Constraint Decoder** (`constraint_decoder.py`)
   - Integrates with HuggingFace models
   - Real-time token masking during generation
   - State management and transitions

5. **JSON Grammar Helper** (`json_grammar.py`)
   - Predefined JSON grammar
   - Simplified interface for JSON generation

## Key Features Implemented

✅ **CFG Parsing**: Parse grammar rules and build grammar structures  
✅ **FSM Support**: Finite State Machine for simple rules  
✅ **PDA Support**: Pushdown Automaton for stack-based rules  
✅ **Token Masking**: Mask invalid tokens during generation  
✅ **Model Integration**: Works with HuggingFace models  
✅ **JSON Grammar**: Predefined JSON grammar for testing  

## Implementation Details

### Grammar Parsing

```python
rules = [
    'value -> "true"',
    'value -> "false"',
]
grammar = Grammar(rules, start_symbol="value")
```

- Parses rules in format: `lhs -> rhs`
- Handles terminals (quoted strings) and non-terminals
- Detects recursive rules to determine stack requirement

### Token Masking

The constraint decoder masks invalid tokens by:
1. Getting valid tokens from current FSM/PDA state
2. Converting token strings to token IDs
3. Setting logits of invalid tokens to -inf
4. Sampling from masked distribution

### State Transitions

- **FSM**: Simple state transitions based on current state
- **PDA**: Stack-based transitions for nested structures
- State updated after each token generation

## Test Results

The test suite (`test_constraint_decoding.py`) includes:

1. **Grammar Information Test**: Verifies grammar parsing and automaton creation
2. **Boolean Grammar Test**: Simple FSM-based constraint (true/false)
3. **Array Grammar Test**: PDA-based constraint for JSON arrays
4. **JSON Object Test**: Full JSON grammar with PDA

## Comparison with SGLang

| Feature | This Implementation | SGLang X-Grammar |
|---------|---------------------|------------------|
| CFG Support | ✅ Basic | ✅ Full |
| FSM | ✅ | ✅ |
| PDA | ✅ Basic | ✅ Advanced |
| Token Masking | ✅ | ✅ Optimized |
| Tree-based Stack | ❌ | ✅ |
| State Inlining | ❌ | ✅ |
| GPU Overlap | ❌ | ✅ |
| Performance | Python overhead | Optimized C++/CUDA |

## Limitations

This is a **simplified educational implementation**. Key limitations:

1. **Token Matching**: Simplified token-to-ID mapping (real implementation uses sophisticated matching)
2. **Grammar Compilation**: Basic CFG parsing (real implementation has advanced optimizations)
3. **Stack Management**: Simple stack operations (real implementation uses tree-based management)
4. **Performance**: Python overhead (real implementation uses optimized kernels)

## Usage Example

```python
from constraint_decoder import ConstraintDecoder
from json_grammar import JSONGrammar

# Create decoder
decoder = ConstraintDecoder(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda"
)

# Set JSON grammar
json_grammar = JSONGrammar()
decoder.set_grammar(json_grammar.get_grammar())

# Generate constrained output
prompt = "Generate a JSON object with name, age, and city fields:"
generated = decoder.generate(prompt, max_new_tokens=50)
print(generated)
```

## Files

- `grammar.py`: CFG parser and compiler
- `fsm.py`: Finite State Machine implementation
- `pda.py`: Pushdown Automaton implementation
- `constraint_decoder.py`: Main constraint decoder
- `json_grammar.py`: JSON grammar helper
- `test_constraint_decoding.py`: Test suite
- `README.md`: Documentation

## Conclusion

This implementation demonstrates the core concepts of SGLang's X-Grammar:
- Grammar-based constraint decoding
- FSM for simple rules
- PDA for stack-based rules
- Token masking during generation

While simplified compared to the production SGLang implementation, it provides a clear educational example of how constraint decoding works.
