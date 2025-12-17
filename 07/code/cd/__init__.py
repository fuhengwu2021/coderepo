"""
Constraint Decoding (X-Grammar) implementation for SGLang-style structured output.

This module implements a simplified version of SGLang's X-Grammar,
which enables LLMs to generate outputs that conform to specific formats
using Context-Free Grammars (CFG), Finite State Machines (FSMs), and
Pushdown Automata (PDAs).
"""

from .grammar import Grammar, GrammarRule
from .fsm import FSM, FSMState
from .pda import PDA, PDAState
from .constraint_decoder import ConstraintDecoder
from .json_grammar import JSONGrammar

__all__ = [
    "Grammar",
    "GrammarRule",
    "FSM",
    "FSMState",
    "PDA",
    "PDAState",
    "ConstraintDecoder",
    "JSONGrammar",
]
