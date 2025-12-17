# Constraint Decoding (X-Grammar) Implementation

A simplified Python implementation of SGLang's X-Grammar constraint decoding system for generating structured outputs that conform to Context-Free Grammars (CFG).

## Overview

This implementation provides:

- **Grammar Parser**: Parses CFG rules and builds grammar structures
- **Finite State Machine (FSM)**: For simple context-free rules
- **Pushdown Automaton (PDA)**: For stack-based rules (nested structures)
- **Constraint Decoder**: Integrates with HuggingFace models to generate grammar-conforming outputs
- **Token Masking**: Masks invalid tokens during generation

## Features

### 1. Grammar Support
- Context-Free Grammar (CFG) parsing
- Automatic detection of stack requirements
- Support for terminals, non-terminals, and epsilon rules

### 2. Automata
- **FSM**: For simple rules where token validity depends only on current state
- **PDA**: For rules requiring stack state (nested structures, balanced parentheses)

### 3. Constraint Decoding
- Real-time token masking during generation
- Integration with HuggingFace models
- Temperature and top-p sampling support

## File Structure

```
cd/
├── __init__.py              # Package initialization
├── grammar.py               # CFG parser and compiler
├── fsm.py                   # Finite State Machine implementation
├── pda.py                   # Pushdown Automaton implementation
├── constraint_decoder.py    # Main constraint decoder
├── json_grammar.py          # JSON grammar helper
├── test_constraint_decoding.py  # Test suite
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Usage

### Basic Example

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

### Custom Grammar

```python
from constraint_decoder import ConstraintDecoder
from grammar import Grammar

# Define custom grammar
rules = [
    'value -> "true"',
    'value -> "false"',
]
grammar = Grammar(rules, start_symbol="value")

# Create decoder with grammar
decoder = ConstraintDecoder(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda",
    grammar=grammar
)

# Generate
generated = decoder.generate("Answer true or false: The sky is blue.")
```

## Running Tests

```bash
conda activate usao
cd chapter7-request-level-routing-and-sglang/code/cd
python test_constraint_decoding.py
```

## Implementation Details

### Grammar Parsing
- Parses CFG rules in format: `lhs -> rhs`
- Handles terminals (quoted strings) and non-terminals
- Detects recursive rules to determine if stack is needed

### Token Masking
- Precompiles valid tokens for each state
- Masks invalid tokens by setting logits to -inf
- Handles tokenizer variations (spaces, quotes, etc.)

### State Management
- FSM: Simple state transitions based on current state
- PDA: Stack-based transitions for nested structures
- Automatic state updates after each token generation

## Limitations

This is a **simplified educational implementation**. Compared to SGLang's production X-Grammar:

1. **Token Matching**: Simplified token-to-ID mapping (real implementation uses more sophisticated matching)
2. **Grammar Compilation**: Basic CFG parsing (real implementation has advanced optimizations)
3. **Stack Management**: Simple stack operations (real implementation uses tree-based management)
4. **Performance**: Python overhead (real implementation uses optimized C++/CUDA)

## Comparison with SGLang

| Feature | This Implementation | SGLang X-Grammar |
|---------|---------------------|------------------|
| CFG Support | ✅ Basic | ✅ Full |
| FSM | ✅ | ✅ |
| PDA | ✅ Basic | ✅ Advanced |
| Token Masking | ✅ | ✅ Optimized |
| Tree-based Stack | ❌ | ✅ |
| GPU Overlap | ❌ | ✅ |
| State Inlining | ❌ | ✅ |

## References

- SGLang X-Grammar: `python/sglang/srt/constrained/xgrammar_backend.py`
- Chapter 7: Structured Output Decoding with X-Grammar
