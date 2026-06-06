"""
Mini APL parser for pruning formulas.
Translates a minimal APL subset into executable JAX code.
"""

import jax.numpy as jnp
from typing import Dict, Any


class MiniAPLParser:
    """
    Parse a minimal APL subset for scoring formulas.

    Supported primitives:
        |X|        : absolute value
        mean(X)    : mean
        var(X)     : variance
        norm(X)    : L2 norm
        sum(X)     : sum
        max(X)     : maximum
        min(X)     : minimum
        sqrt(X)    : square root
        log(X)     : natural log
        exp(X)     : exponential
        X + Y      : addition
        X - Y      : subtraction
        X x Y      : multiplication
        X / Y      : division
        X ^ Y      : power
        scores <- expr : assignment
    """

    def __init__(self):
        self.variables: Dict[str, Any] = {}

    def set_variable(self, name: str, value):
        """Bind a JAX tensor to an APL variable name."""
        self.variables[name] = value

    def set_variables(self, **kwargs):
        """Bind multiple variables at once."""
        self.variables.update(kwargs)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, code: str) -> list:
        """Split an APL expression into tokens."""
        tokens = []
        i = 0
        while i < len(code):
            c = code[i]

            if c.isspace():
                i += 1
                continue

            # APL comments
            if c == '\u235d':
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue

            # Multi-char symbols
            if code[i:i+2] == '+/':
                tokens.append(('REDUCE_SUM', '+/'))
                i += 2
                continue

            # Single char symbols
            if c in '+x/^':
                tokens.append(('OP', c))
                i += 1
                continue
            if c == '-':
                if not tokens or tokens[-1][0] in ('OP', 'LPAREN', 'ASSIGN'):
                    tokens.append(('NEG', c))
                else:
                    tokens.append(('OP', c))
                i += 1
                continue
            if c == '|':
                tokens.append(('ABS', c))
                i += 1
                continue
            if c == '(':
                tokens.append(('LPAREN', c))
                i += 1
                continue
            if c == ')':
                tokens.append(('RPAREN', c))
                i += 1
                continue
            if c == '\u2190':
                tokens.append(('ASSIGN', c))
                i += 1
                continue

            # Words (functions, variables)
            if c.isalpha() or c == '_':
                start = i
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    i += 1
                word = code[start:i]
                if word in ('mean', 'var', 'norm', 'sum', 'max', 'min',
                            'sqrt', 'log', 'exp', 'abs'):
                    tokens.append(('FUNC', word))
                else:
                    tokens.append(('VAR', word))
                continue

            # Numbers
            if c.isdigit() or c == '.':
                start = i
                while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                    i += 1
                tokens.append(('NUM', float(code[start:i])))
                continue

            i += 1

        return tokens

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_expression(self, tokens: list, pos: int) -> tuple:
        """Parse an APL expression into an AST."""

        def parse_primary():
            nonlocal pos
            if pos >= len(tokens):
                return None

            tok_type, tok_val = tokens[pos]

            if tok_type == 'NEG':
                pos += 1
                operand = parse_primary()
                return ('neg', operand)

            if tok_type == 'ABS':
                pos += 1
                operand = parse_primary()
                if pos < len(tokens) and tokens[pos][0] == 'ABS':
                    pos += 1
                return ('abs', operand)

            if tok_type == 'LPAREN':
                pos += 1
                ast = parse_expression()
                if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                    pos += 1
                return ast

            if tok_type == 'FUNC':
                func_name = tok_val
                pos += 1
                if pos < len(tokens) and tokens[pos][0] == 'LPAREN':
                    pos += 1
                    arg = parse_expression()
                    if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                        pos += 1
                    return (func_name, arg)
                else:
                    arg = parse_primary()
                    return (func_name, arg)

            if tok_type == 'REDUCE_SUM':
                pos += 1
                arg = parse_primary()
                return ('sum', arg)

            if tok_type == 'VAR':
                pos += 1
                return ('var', tok_val)

            if tok_type == 'NUM':
                pos += 1
                return ('num', tok_val)

            pos += 1
            return None

        def parse_binary(left, min_precedence=0):
            nonlocal pos
            precedence = {'+': 1, '-': 1, 'x': 2, '/': 2, '^': 3}
            op_map = {'+': 'add', '-': 'sub', 'x': 'mul', '/': 'div', '^': 'pow'}

            while pos < len(tokens) and tokens[pos][0] == 'OP':
                op = tokens[pos][1]
                prec = precedence.get(op, 0)
                if prec < min_precedence:
                    break
                pos += 1
                right = parse_primary()
                if pos < len(tokens) and tokens[pos][0] == 'OP':
                    next_prec = precedence.get(tokens[pos][1], 0)
                    if next_prec > prec:
                        right = parse_binary(right, next_prec)
                left = (op_map[op], left, right)

            return left

        ast = parse_primary()
        ast = parse_binary(ast)
        return (ast, pos)

    def parse_expression(self, code: str):
        """Parse a single APL expression."""
        tokens = self._tokenize(code)
        ast, _ = self._parse_expression(tokens, 0)
        return ast

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------

    def _eval_ast(self, ast) -> Any:
        """Evaluate an AST using JAX."""
        if ast is None:
            return None

        node_type = ast[0]

        if node_type == 'num':
            return float(ast[1])

        if node_type == 'var':
            var_name = ast[1]
            if var_name not in self.variables:
                raise NameError(
                    f"Variable '{var_name}' not defined. "
                    f"Available variables: {list(self.variables.keys())}"
                )
            return self.variables[var_name]

        if node_type == 'neg':
            return -self._eval_ast(ast[1])

        if node_type == 'abs':
            return jnp.abs(self._eval_ast(ast[1]))

        if node_type == 'mean':
            return jnp.mean(self._eval_ast(ast[1]))

        if node_type == 'var':
            return jnp.var(self._eval_ast(ast[1]))

        if node_type == 'norm':
            return jnp.linalg.norm(self._eval_ast(ast[1]))

        if node_type == 'sum':
            return jnp.sum(self._eval_ast(ast[1]))

        if node_type == 'max':
            return jnp.max(self._eval_ast(ast[1]))

        if node_type == 'min':
            return jnp.min(self._eval_ast(ast[1]))

        if node_type == 'sqrt':
            return jnp.sqrt(self._eval_ast(ast[1]))

        if node_type == 'log':
            return jnp.log(self._eval_ast(ast[1]))

        if node_type == 'exp':
            return jnp.exp(self._eval_ast(ast[1]))

        if node_type == 'add':
            return self._eval_ast(ast[1]) + self._eval_ast(ast[2])

        if node_type == 'sub':
            return self._eval_ast(ast[1]) - self._eval_ast(ast[2])

        if node_type == 'mul':
            return self._eval_ast(ast[1]) * self._eval_ast(ast[2])

        if node_type == 'div':
            return self._eval_ast(ast[1]) / self._eval_ast(ast[2])

        if node_type == 'pow':
            return self._eval_ast(ast[1]) ** self._eval_ast(ast[2])

        raise ValueError(f"Unknown AST node type: {node_type}")

    # ------------------------------------------------------------------
    # Compilation and execution
    # ------------------------------------------------------------------

    def compile(self, code: str):
        """
        Compile APL code (possibly multi-line with assignments)
        and return a callable function.
        """
        lines = [
            l.strip() for l in code.strip().split('\n')
            if l.strip() and not l.strip().startswith('\u235d')
        ]

        if len(lines) == 1 and '\u2190' not in lines[0]:
            ast = self.parse_expression(lines[0])

            def fn():
                return self._eval_ast(ast)
            return fn

        final_expr = None

        for line in lines:
            if '\u2190' in line:
                parts = line.split('\u2190', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()

                ast = self.parse_expression(expr)
                result = self._eval_ast(ast)
                self.variables[var_name] = result
            else:
                final_expr = self.parse_expression(line)

        def fn():
            return self._eval_ast(final_expr) if final_expr else None
        return fn

    def evaluate(self, code: str):
        """Compile and execute immediately. Returns the result."""
        fn = self.compile(code)
        return fn()

    def __call__(self, code: str):
        """Syntactic sugar: parser("|W| x mean(|act|)")"""
        return self.evaluate(code)
