"""
Demo script for Constraint Decoding (X-Grammar) implementation.

Demonstrates the constraint decoder with Qwen/Qwen2.5-0.5B-Instruct
using JSON grammar examples.
"""

import torch
import time
from constraint_decoder import ConstraintDecoder
from json_grammar import JSONGrammar
from grammar import Grammar


def simple_json_demo():
    """Demo: Simple JSON object generation."""
    print("=" * 60)
    print("Demo 1: Simple JSON Object Generation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create decoder
    decoder = ConstraintDecoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device
    )
    
    # Set JSON grammar
    json_grammar = JSONGrammar()
    decoder.set_grammar(json_grammar.get_grammar())
    
    # Test prompt
    prompt = "Generate a JSON object with name, age, and city fields:"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating constrained output...")
    
    start_time = time.time()
    generated = decoder.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nGenerated text:")
    print(generated)
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    print()


def boolean_grammar_demo():
    """Demo: Simple boolean grammar."""
    print("=" * 60)
    print("Demo 2: Boolean Grammar (FSM)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create simple boolean grammar
    rules = [
        'value -> "true"',
        'value -> "false"',
    ]
    grammar = Grammar(rules, start_symbol="value")
    
    # Create decoder
    decoder = ConstraintDecoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        grammar=grammar
    )
    
    prompt = "Answer with true or false: The sky is blue."
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating constrained output (must be 'true' or 'false')...")
    
    start_time = time.time()
    generated = decoder.generate(
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.3,
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nGenerated text:")
    print(generated)
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    print()


def array_grammar_demo():
    """Demo: Array grammar."""
    print("=" * 60)
    print("Demo 3: JSON Array Grammar")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create array grammar
    rules = [
        'array -> "[" elements "]"',
        'array -> "[" "]"',
        'elements -> value',
        'elements -> value "," elements',
        'value -> string',
        'value -> number',
        'string -> "\\"" chars "\\""',
        'chars -> char chars',
        'chars -> ε',
        'number -> digit number',
        'number -> digit',
    ]
    grammar = Grammar(rules, start_symbol="array")
    
    # Create decoder
    decoder = ConstraintDecoder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        grammar=grammar
    )
    
    prompt = "Generate a JSON array of three city names:"
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating constrained output (must be valid JSON array)...")
    
    start_time = time.time()
    generated = decoder.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.7,
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nGenerated text:")
    print(generated)
    print(f"\nTime taken: {elapsed_time:.2f} seconds")
    print()


def grammar_info_demo():
    """Demo: Grammar parsing and automaton creation."""
    print("=" * 60)
    print("Demo 4: Grammar Information")
    print("=" * 60)
    
    json_grammar = JSONGrammar()
    grammar = json_grammar.get_grammar()
    
    print(f"\nGrammar start symbol: {grammar.start_symbol}")
    print(f"Requires stack: {grammar.requires_stack}")
    print(f"Non-terminals: {sorted(grammar.non_terminals)}")
    print(f"Terminals: {sorted(list(grammar.terminals))[:20]}...")  # Show first 20
    
    print(f"\nSample rules:")
    for non_terminal in list(grammar.non_terminals)[:5]:
        rules = grammar.get_rules(non_terminal)
        for rule in rules[:2]:  # Show first 2 rules
            print(f"  {rule}")
    
    # Test FSM/PDA creation
    if grammar.requires_stack:
        print("\nUsing PDA (Pushdown Automaton)")
        from pda import PDA
        pda = PDA(grammar, start_symbol=grammar.start_symbol)
        print(f"PDA states: {list(pda.states.keys())[:5]}...")
    else:
        print("\nUsing FSM (Finite State Machine)")
        from fsm import FSM
        fsm = FSM(grammar, start_symbol=grammar.start_symbol)
        print(f"FSM states: {list(fsm.states.keys())[:5]}...")
        print(f"Valid tokens in start state: {list(fsm.get_valid_tokens())[:10]}")
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Constraint Decoding (X-Grammar) Demo Suite")
    print("=" * 60)
    print()
    
    try:
        # Demo grammar info first
        grammar_info_demo()
        
        # Demo simple boolean grammar (fastest)
        boolean_grammar_demo()
        
        # Demo array grammar
        array_grammar_demo()
        
        # Demo full JSON grammar
        simple_json_demo()
        
        print("=" * 60)
        print("All demos completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
