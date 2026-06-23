"""Main parser class: compiles APL code to executable functions."""

from typing import Dict, Any

from apl_pruning.tokenizer import tokenize
from apl_pruning.grammar import parse_expression
from apl_pruning.ast_evaluator import eval_ast


class MiniAPLParser:
    """Parse a minimal APL subset for scoring formulas.
    
    Usage:
        parser = MiniAPLParser()
        parser.set_variables(W=weights, act=activations)
        scores = parser.evaluate("|W| x mean(|act|)")
    """
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
    
    def set_variable(self, name: str, value):
        self.variables[name] = value
    
    def set_variables(self, **kwargs):
        self.variables.update(kwargs)
    
    def parse(self, code: str):
        """Parse APL code into an AST."""
        tokens = tokenize(code)
        ast, _ = parse_expression(tokens, 0)
        return ast
    
    def compile(self, code: str):
        """Compile APL code into a callable function."""
        lines = [
            l.strip() for l in code.strip().split('\n')
            if l.strip()
            and not l.strip().startswith('\u235d')
            and not l.strip().startswith('//')
        ]
        
        # Single expression
        if len(lines) == 1 and '<-' not in lines[0] and '\u2190' not in lines[0]:
            ast = self.parse(lines[0])
            def fn():
                return eval_ast(ast, self.variables)
            return fn
        
        # Multi-line with assignments
        final_ast = None
        for line in lines:
            var_name, expr = self._split_assignment(line)
            if var_name:
                ast = self.parse(expr)
                result = eval_ast(ast, self.variables)
                self.variables[var_name] = result
            else:
                final_ast = self.parse(line)
        
        def fn():
            return eval_ast(final_ast, self.variables) if final_ast else None
        return fn
    
    def evaluate(self, code: str):
        """Compile and execute immediately."""
        return self.compile(code)()
    
    def to_pytorch(self, code: str) -> str:
        """Export APL expression to PyTorch code.
        
        Example:
            >>> parser.to_pytorch("|W| x mean(|act|)")
            'torch.abs(W) * torch.mean(torch.abs(act))'
        """
        from apl_pruning.exporter import to_pytorch
        ast = self.parse(code)
        return to_pytorch(ast)
    
    def to_pytorch_function(self, code: str, func_name: str = None) -> str:
        """Export APL code as a complete PyTorch function.
        
        Example:
            >>> print(parser.to_pytorch_function("|W| x mean(|act|)", "wanda"))
            import torch
            def wanda(W, act):
                return torch.abs(W) * torch.mean(torch.abs(act))
        """
        from apl_pruning.exporter import to_pytorch_function
        return to_pytorch_function(code, func_name)
    
    def __call__(self, code: str):
        return self.evaluate(code)
    
    @staticmethod
    def _split_assignment(line: str):
        """Split 'name <- expr' into (name, expr)."""
        for arrow in ('<-', '\u2190'):
            if arrow in line:
                parts = line.split(arrow, 1)
                return parts[0].strip(), parts[1].strip()
        return None, None
