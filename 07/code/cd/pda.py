"""
Pushdown Automaton (PDA) for context-dependent grammar rules.

PDAs extend FSMs with a stack to handle nested structures
and context-dependent validation (e.g., balanced parentheses).
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
try:
    from .fsm import FSM, FSMState
except ImportError:
    from fsm import FSM, FSMState


@dataclass
class PDAState:
    """State in a Pushdown Automaton."""
    name: str
    is_accepting: bool = False
    transitions: Dict[Tuple[str, Optional[str]], Tuple[str, List[str]]] = None
    # (input_token, stack_top) -> (next_state, stack_operations)
    
    def __post_init__(self):
        if self.transitions is None:
            self.transitions = {}


class PDA:
    """
    Pushdown Automaton for constraint decoding with stack.
    
    Used for grammar rules that require stack state, such as
    nested structures and balanced parentheses.
    """
    
    def __init__(self, grammar, start_symbol: str = "S"):
        """
        Build PDA from grammar.
        
        Args:
            grammar: Grammar object
            start_symbol: Start symbol
        """
        self.grammar = grammar
        self.start_symbol = start_symbol
        self.states: Dict[str, PDAState] = {}
        self.current_state = start_symbol
        self.stack: List[str] = []  # Stack for context
        
        # Build PDA states from grammar rules
        self._build_pda()
        
        # Precompile valid tokens (simplified - doesn't account for stack)
        self.valid_tokens_cache: Dict[str, Set[str]] = {}
        self._precompile_tokens()
    
    def _build_pda(self):
        """Build PDA states from grammar rules."""
        # Create states for each non-terminal
        for non_terminal in self.grammar.non_terminals:
            state = PDAState(name=non_terminal)
            self.states[non_terminal] = state
        
        # Build transitions with stack operations
        for non_terminal, rules in self.grammar.rules.items():
            state = self.states[non_terminal]
            
            for rule in rules:
                if len(rule.rhs) == 0:  # Epsilon rule
                    state.is_accepting = True
                else:
                    # For recursive rules, push/pop stack
                    if non_terminal in rule.rhs:
                        # Push current non-terminal
                        key = (rule.rhs[0], None)  # First symbol, any stack
                        state.transitions[key] = (non_terminal, [non_terminal])  # Push
                    else:
                        # Regular transition
                        first_symbol = rule.rhs[0] if rule.rhs else None
                        if first_symbol:
                            key = (first_symbol, None)
                            next_state = rule.rhs[1] if len(rule.rhs) > 1 else None
                            state.transitions[key] = (next_state or non_terminal, [])
    
    def _precompile_tokens(self):
        """Precompile valid tokens (simplified - doesn't account for stack state)."""
        for state_name, state in self.states.items():
            valid_tokens = set()
            
            for (token, _), _ in state.transitions.items():
                if self.grammar.is_terminal(token):
                    valid_tokens.add(token)
            
            self.valid_tokens_cache[state_name] = valid_tokens
    
    def get_valid_tokens(self, state: Optional[str] = None, stack_top: Optional[str] = None) -> Set[str]:
        """
        Get valid tokens for a state and stack configuration.
        
        Args:
            state: State name (None = current state)
            stack_top: Top of stack (None = any)
            
        Returns:
            Set of valid token strings
        """
        state_name = state or self.current_state
        
        # Check transitions that match stack_top
        valid_tokens = set()
        state_obj = self.states.get(state_name)
        if state_obj:
            for (token, required_stack_top), _ in state_obj.transitions.items():
                if required_stack_top is None or required_stack_top == stack_top:
                    if self.grammar.is_terminal(token):
                        valid_tokens.add(token)
        
        return valid_tokens
    
    def is_valid_token(self, token: str, state: Optional[str] = None, stack_top: Optional[str] = None) -> bool:
        """Check if a token is valid given state and stack."""
        return token in self.get_valid_tokens(state, stack_top)
    
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
        
        stack_top = self.stack[-1] if self.stack else None
        
        # Try to find matching transition
        for (input_token, required_stack_top), (next_state, stack_ops) in state.transitions.items():
            if input_token == token:
                if required_stack_top is None or required_stack_top == stack_top:
                    # Execute stack operations
                    for op in stack_ops:
                        if op == "pop":
                            if self.stack:
                                self.stack.pop()
                        else:
                            # Push operation
                            self.stack.append(op)
                    
                    if next_state:
                        self.current_state = next_state
                        return next_state
                    return None
        
        return None
    
    def reset(self):
        """Reset PDA to initial state."""
        self.current_state = self.start_symbol
        self.stack = []
    
    def is_accepting(self, state: Optional[str] = None) -> bool:
        """Check if current state is accepting."""
        state_name = state or self.current_state
        state_obj = self.states.get(state_name)
        return state_obj.is_accepting if state_obj else False
