"""
JSON grammar helper for easy JSON constraint decoding.
"""

try:
    from .grammar import Grammar, parse_json_grammar
except ImportError:
    from grammar import Grammar, parse_json_grammar


class JSONGrammar:
    """
    Helper class for JSON constraint decoding.
    
    Provides a simplified interface for generating JSON outputs.
    """
    
    def __init__(self):
        """Initialize JSON grammar."""
        self.grammar = parse_json_grammar()
    
    def get_grammar(self) -> Grammar:
        """Get the underlying Grammar object."""
        return self.grammar
    
    @staticmethod
    def create_simple_grammar() -> Grammar:
        """
        Create a simplified JSON grammar for basic objects.
        
        Returns:
            Grammar object
        """
        rules = [
            'object -> "{" pairs "}"',
            'object -> "{" "}"',
            'pairs -> pair',
            'pairs -> pair "," pairs',
            'pair -> string ":" value',
            'value -> string',
            'value -> number',
            'value -> boolean',
            'value -> null',
            'value -> object',
            'string -> "\\"" chars "\\""',
            'chars -> char chars',
            'chars -> ε',
            'number -> digit number',
            'number -> digit',
            'boolean -> "true"',
            'boolean -> "false"',
            'null -> "null"',
        ]
        return Grammar(rules, start_symbol="object")
