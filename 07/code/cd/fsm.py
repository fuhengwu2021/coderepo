"""
Finite State Machine (FSM) for simple context-free grammar rules.

FSMs are used for rules where token validity depends only on
the current state, not on stack/history.
"""

from typing import Dict, Set, List, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FSMState:
    """State in a Finite State Machine."""
    name: str
    is_accepting: bool = False
    transitions: Dict[str, str] = None  # token -> next_state
    
    def __post_init__(self):
        if self.transitions is None:
            self.transitions = {}


class FSM:
    """
    Finite State Machine for constraint decoding.
    
    Used for simple grammar rules that don't require stack state.
    Precompiles valid tokens for each state for efficient lookup.
    """
    
    def __init__(self, grammar, start_symbol: str = "S"):
        """
        Build FSM from grammar.
        
        Args:
            grammar: Grammar object
            start_symbol: Start symbol
        """
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.states: Dict[str, FSMState] = {}
        self.current_state = start_symbol
        
        # Build FSM states from grammar rules
        self._build_fsm()
        
        # Precompile valid tokens for each state
        self.valid_tokens_cache: Dict[str, Set[str]] = {}
        self._precompile_tokens()
    
    def _build_fsm(self):
        """Build FSM states from grammar rules."""
        # Create states for each non-terminal
        for non_terminal in self.grammar.non_terminals:
            state = FSMState(name=non_terminal)
            self.states[non_terminal] = state
        
        # Build transitions
        for non_terminal, rules in self.grammar.rules.items():
            state = self.states[non_terminal]
            
            for rule in rules:
                # For each rule, create transitions
                if len(rule.rhs) == 0:  # Epsilon rule
                    state.is_accepting = True
                elif len(rule.rhs) == 1:  # Single symbol
                    symbol = rule.rhs[0]
                    if self.grammar.is_terminal(symbol):
                        # Terminal transition
                        if symbol not in state.transitions:
                            state.transitions[symbol] = None  # Accepting state
                    else:
                        # Non-terminal transition
                        state.transitions[symbol] = symbol
                else:
                    # Multiple symbols - simplified: take first terminal
                    for symbol in rule.rhs:
                        if self.grammar.is_terminal(symbol):
                            if symbol not in state.transitions:
                                state.transitions[symbol] = None
                            break
    
    def _precompile_tokens(self):
        """Precompile valid tokens for each state."""
        for state_name, state in self.states.items():
            valid_tokens = set()
            
            # Get all terminal transitions from this state
            for token, next_state in state.transitions.items():
                if self.grammar.is_terminal(token):
                    valid_tokens.add(token)
            
            # Also check rules that start with terminals
            if state_name in self.grammar.rules:
                for rule in self.grammar.rules[state_name]:
                    if rule.rhs and self.grammar.is_terminal(rule.rhs[0]):
                        valid_tokens.add(rule.rhs[0])
            
            self.valid_tokens_cache[state_name] = valid_tokens
    
    def get_valid_tokens(self, state: Optional[str] = None) -> Set[str]:
        """
        Get valid tokens for a state.
        
        Args:
            state: State name (None = current state)
            
        Returns:
            Set of valid token strings
        """
        state_name = state or self.current_state
        return self.valid_tokens_cache.get(state_name, set())
    
    def is_valid_token(self, token: str, state: Optional[str] = None) -> bool:
        """Check if a token is valid in the current state."""
        return token in self.get_valid_tokens(state)
    
    def transition(self, token: str) -> Optional[str]:
        """
        Make a transition with a token.
        
        Args:
            token: Input token
            
        Returns:
            Next state name, or None if invalid transition
        """
        state = self.states.get(self.current_state)
        if not state:
            return None
        
        if token in state.transitions:
            next_state = state.transitions[token]
            if next_state is None:
                # Accepting state
                return None
            self.current_state = next_state
            return next_state
        
        return None
    
    def reset(self):
        """Reset FSM to initial state."""
        self.current_state = self.start_symbol
    
    def is_accepting(self, state: Optional[str] = None) -> bool:
        """Check if current state is accepting."""
        state_name = state or self.current_state
        state_obj = self.states.get(state_name)
        return state_obj.is_accepting if state_obj else False
