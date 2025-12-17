"""
Grammar parser and compiler for Context-Free Grammars (CFG).

This module parses CFG rules and compiles them into structures
that can be used for constraint decoding.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import re


@dataclass
class GrammarRule:
    """Represents a single grammar rule."""
    lhs: str  # Left-hand side (non-terminal)
    rhs: List[str]  # Right-hand side (sequence of terminals/non-terminals)
    is_terminal: bool = False  # True if this is a terminal symbol
    
    def __repr__(self):
        rhs_str = " ".join(self.rhs)
        return f"{self.lhs} -> {rhs_str}"


class Grammar:
    """
    Context-Free Grammar parser and compiler.
    
    Supports parsing CFG rules and determining if a grammar
    requires stack-based processing (PDA) or can use FSM.
    """
    
    def __init__(self, rules: List[str], start_symbol: str = "S"):
        """
        Initialize grammar from rule strings.
        
        Args:
            rules: List of rule strings, e.g., ['S -> "a" S "b"', 'S -> ε']
            start_symbol: Start symbol of the grammar
        """
        self.start_symbol = start_symbol
        self.rules: Dict[str, List[GrammarRule]] = {}
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = {start_symbol}
        
        # Parse rules
        for rule_str in rules:
            self._parse_rule(rule_str)
        
        # Determine if grammar needs stack (PDA) or can use FSM
        self.requires_stack = self._requires_stack()
    
    def _parse_rule(self, rule_str: str):
        """Parse a single rule string."""
        # Remove comments and whitespace
        rule_str = rule_str.strip()
        if not rule_str or rule_str.startswith("#"):
            return
        
        # Split on ->
        if "->" not in rule_str:
            return
        
        parts = rule_str.split("->", 1)
        lhs = parts[0].strip()
        rhs_str = parts[1].strip()
        
        # Parse right-hand side
        # Handle terminals (quoted strings) and non-terminals
        rhs = []
        # Simple tokenizer: split by whitespace, handle quoted strings
        tokens = re.findall(r'"[^"]*"|\S+', rhs_str)
        
        for token in tokens:
            if token.startswith('"') and token.endswith('"'):
                # Terminal symbol
                terminal = token[1:-1]  # Remove quotes
                rhs.append(terminal)
                self.terminals.add(terminal)
            elif token == "ε" or token == "epsilon":
                # Epsilon (empty string)
                rhs.append("")
            else:
                # Non-terminal
                rhs.append(token)
                self.non_terminals.add(token)
        
        # Add rule
        rule = GrammarRule(lhs=lhs, rhs=rhs)
        if lhs not in self.rules:
            self.rules[lhs] = []
        self.rules[lhs].append(rule)
    
    def _requires_stack(self) -> bool:
        """
        Determine if grammar requires stack-based processing.
        
        A grammar needs a stack if:
        1. It has recursive rules (A -> ... A ...)
        2. It has nested structures (e.g., balanced parentheses)
        3. It has context-dependent rules
        
        For simplicity, we check for recursive rules.
        """
        # Check for direct recursion
        for lhs, rule_list in self.rules.items():
            for rule in rule_list:
                if lhs in rule.rhs:
                    return True
        
        # Check for indirect recursion (simplified - just check one level)
        for lhs, rule_list in self.rules.items():
            for rule in rule_list:
                for symbol in rule.rhs:
                    if symbol in self.non_terminals and symbol != lhs:
                        # Check if symbol can derive lhs (simplified check)
                        if symbol in self.rules:
                            for sub_rule in self.rules[symbol]:
                                if lhs in sub_rule.rhs:
                                    return True
        
        return False
    
    def get_rules(self, non_terminal: str) -> List[GrammarRule]:
        """Get all rules for a non-terminal."""
        return self.rules.get(non_terminal, [])
    
    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is a terminal."""
        return symbol in self.terminals
    
    def is_non_terminal(self, symbol: str) -> bool:
        """Check if a symbol is a non-terminal."""
        return symbol in self.non_terminals


def parse_json_grammar() -> Grammar:
    """Parse a simplified JSON grammar."""
    rules = [
        'value -> object',
        'value -> array',
        'value -> string',
        'value -> number',
        'value -> boolean',
        'value -> null',
        'object -> "{" pairs "}"',
        'object -> "{" "}"',
        'pairs -> pair',
        'pairs -> pair "," pairs',
        'pair -> string ":" value',
        'array -> "[" elements "]"',
        'array -> "[" "]"',
        'elements -> value',
        'elements -> value "," elements',
        'string -> "\\"" chars "\\""',
        'chars -> char chars',
        'chars -> ε',
        'number -> digit number',
        'number -> digit',
        'boolean -> "true"',
        'boolean -> "false"',
        'null -> "null"',
    ]
    return Grammar(rules, start_symbol="value")
