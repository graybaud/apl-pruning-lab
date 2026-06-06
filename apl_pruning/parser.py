"""
Mini APL parser for pruning formulas - Level 2.
Translates a minimal APL subset into executable JAX code.

New in v0.2.0:
    - Axis support: mean(X, dim=-1), sum(X, dim=0), etc.
    - Indexing: W[i], W[:, j], W[0:512]
    - New primitives: std, softmax, topk, threshold, where
    - Broadcasting-aware operations
    - Better error messages with shape info
"""

import jax.numpy as jnp
import jax
import re
from typing import Dict, Any, Optional, Tuple, List


class MiniAPLParser:
    """
    Parse a minimal APL subset for scoring formulas.

    Supported primitives:
        |X|           : absolute value
        mean(X)       : mean (all elements or along axis)
        var(X)        : variance
        std(X)        : standard deviation
        norm(X)       : L2 norm (all elements or along axis)
        sum(X)        : sum
        max(X)        : maximum
        min(X)        : minimum
        sqrt(X)       : square root
        log(X)        : natural log
        exp(X)        : exponential
        softmax(X)    : softmax
        topk(X, k)    : top-k values
        threshold(X,t): binary mask where X > t
        where(cond)   : indices where condition is true

        X + Y         : addition
        X - Y         : subtraction
        X x Y         : multiplication
        X / Y         : division
        X ^ Y         : power

        X[i]          : index row i
        X[i, j]       : index element (i,j)
        X[:, j]       : slice column j
        X[start:end]  : slice rows from start to end

        scores <- expr : assignment

    Axis syntax:
        mean(X, dim=-1)   : mean along last axis
        sum(X, dim=0)     : sum along first axis
        norm(X, dim=-1)   : norm along last axis (per-vector)
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
            if code[i:i+2] == '//':
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue

            # Slice with colon X[start:end]
            if c == ':':
                tokens.append(('COLON', c))
                i += 1
                continue

            # Comma
            if c == ',':
                tokens.append(('COMMA', c))
                i += 1
                continue

            # Equals for kwargs
            if c == '=':
                tokens.append(('EQUALS', c))
                i += 1
                continue

            # Single char symbols
            if c in '+x/^':
                tokens.append(('OP', c))
                i += 1
                continue
            if c == '-':
                if not tokens or tokens[-1][0] in ('OP', 'LPAREN', 'ASSIGN', 'COMMA', 'LBRACKET', 'EQUALS', 'COLON'):
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
            if c == '[':
                tokens.append(('LBRACKET', c))
                i += 1
                continue
            if c == ']':
                tokens.append(('RBRACKET', c))
                i += 1
                continue
            if c == '\u2190':
                tokens.append(('ASSIGN', c))
                i += 1
                continue

            # Words (functions, variables, kwargs)
            if c.isalpha() or c == '_':
                start = i
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    i += 1
                word = code[start:i]
                if word in ('mean', 'var', 'norm', 'sum', 'max', 'min',
                            'sqrt', 'log', 'exp', 'abs', 'std',
                            'softmax', 'topk', 'threshold', 'where',
                            'dim', 'axis'):
                    tokens.append(('FUNC' if word not in ('dim', 'axis') else 'KWARG', word))
                else:
                    tokens.append(('VAR', word))
                continue

            # Numbers
            if c.isdigit() or (c == '.' and i+1 < len(code) and code[i+1].isdigit()):
                start = i
                if c == '-':
                    i += 1
                while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                    i += 1
                num_str = code[start:i]
                tokens.append(('NUM', float(num_str) if '.' in num_str else int(num_str)))
                continue

            i += 1

        return tokens

    # ------------------------------------------------------------------
    # Parser
    # ------------------------------------------------------------------

    def _parse_expression(self, tokens: list, pos: int) -> Tuple[Any, int]:
        """Parse an APL expression into an AST. Returns (ast, new_pos)."""

        def parse_primary():
            nonlocal pos
            if pos >= len(tokens):
                return None

            tok_type, tok_val = tokens[pos]

            # Negation
            if tok_type == 'NEG':
                pos += 1
                operand = parse_primary()
                return ('neg', operand)

            # Absolute value |X|
            if tok_type == 'ABS':
                pos += 1
                operand = parse_primary()
                if pos < len(tokens) and tokens[pos][0] == 'ABS':
                    pos += 1
                return ('abs', operand)

            # Parentheses
            if tok_type == 'LPAREN':
                pos += 1
                ast = parse_expression()
                if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                    pos += 1
                return ast

            # Functions with optional kwargs
            if tok_type == 'FUNC':
                func_name = tok_val
                pos += 1

                # Check for parenthesized args: func(arg1, kwarg=val)
                if pos < len(tokens) and tokens[pos][0] == 'LPAREN':
                    pos += 1  # skip (
                    args = []
                    kwargs = {}

                    while pos < len(tokens) and tokens[pos][0] != 'RPAREN':
                        if tokens[pos][0] == 'COMMA':
                            pos += 1
                            continue

                        # Check if this is a kwarg: dim=0
                        if (pos + 2 < len(tokens) and
                            tokens[pos][0] == 'KWARG' and
                            tokens[pos+1][0] == 'EQUALS'):
                            kw_name = tokens[pos][1]
                            pos += 2  # skip kwarg and =
                            kw_val = parse_primary()
                            kwargs[kw_name] = kw_val
                        else:
                            arg_ast = parse_expression()
                            if arg_ast is not None:
                                args.append(arg_ast)

                        if pos < len(tokens) and tokens[pos][0] == 'COMMA':
                            pos += 1

                    if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                        pos += 1

                    # Build function node with args and kwargs
                    if func_name == 'topk':
                        if len(args) >= 2:
                            return ('topk', args[0], args[1])
                        return ('topk', args[0] if args else None, None)
                    elif func_name == 'threshold':
                        if len(args) >= 2:
                            return ('threshold', args[0], args[1])
                        return ('threshold', args[0] if args else None, None)
                    elif func_name == 'where':
                        return ('where', args[0] if args else None)
                    else:
                        # Functions with axis support
                        axis = kwargs.get('dim', kwargs.get('axis', None))
                        if axis is not None:
                            axis = axis[1] if isinstance(axis, tuple) and axis[0] == 'num' else axis
                        return ('func_with_axis', func_name, args[0] if args else None, axis)
                else:
                    # Function without parentheses: func arg
                    arg = parse_primary()
                    return ('func_with_axis', func_name, arg, None)

            # Reductions
            if tok_type == 'REDUCE_SUM':
                pos += 1
                arg = parse_primary()
                return ('func_with_axis', 'sum', arg, None)

            # Variables with optional indexing
            if tok_type == 'VAR':
                var_name = tok_val
                pos += 1

                # Check for indexing: X[i], X[i, j], X[:, j], X[start:end]
                if pos < len(tokens) and tokens[pos][0] == 'LBRACKET':
                    pos += 1  # skip [
                    indices = []

                    while pos < len(tokens) and tokens[pos][0] != 'RBRACKET':
                        if tokens[pos][0] == 'COMMA':
                            pos += 1
                            continue

                        # Slice with colon: start:end
                        if tokens[pos][0] == 'COLON':
                            # Bare colon means all: X[:, j]
                            start_idx = ('num', 0)
                            pos += 1  # skip :
                            if pos < len(tokens) and tokens[pos][0] != 'RBRACKET' and tokens[pos][0] != 'COMMA':
                                end_idx = parse_primary()
                            else:
                                end_idx = ('num', None)  # None means to the end
                            indices.append(('slice', start_idx, end_idx))
                            continue

                        # Check if next token is colon (start:end)
                        idx_val = parse_primary()
                        if pos < len(tokens) and tokens[pos][0] == 'COLON':
                            pos += 1  # skip :
                            if pos < len(tokens) and tokens[pos][0] != 'RBRACKET' and tokens[pos][0] != 'COMMA':
                                end_val = parse_primary()
                            else:
                                end_val = ('num', None)
                            indices.append(('slice', idx_val, end_val))
                        else:
                            indices.append(idx_val)

                        if pos < len(tokens) and tokens[pos][0] == 'COMMA':
                            pos += 1

                    if pos < len(tokens) and tokens[pos][0] == 'RBRACKET':
                        pos += 1

                    if len(indices) == 1 and not isinstance(indices[0], tuple):
                        return ('index', var_name, indices[0])
                    else:
                        return ('index_multi', var_name, indices)

                return ('var', var_name)

            # Numbers
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
        if ast is not None:
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
            return ast[1]  # Can be int or float

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

        if node_type == 'index':
            var_name = ast[1]
            idx_ast = ast[2]
            var = self.variables[var_name]
            idx = self._eval_ast(idx_ast)
            if isinstance(idx, (int, float)):
                idx = int(idx)
            return var[idx]

        if node_type == 'index_multi':
            var_name = ast[1]
            indices = ast[2]
            var = self.variables[var_name]

            # Build tuple for numpy-style indexing
            idx_tuple = []
            for idx_spec in indices:
                if isinstance(idx_spec, tuple) and idx_spec[0] == 'slice':
                    start = self._eval_ast(idx_spec[1]) if idx_spec[1] else 0
                    end = self._eval_ast(idx_spec[2]) if idx_spec[2] else None
                    if isinstance(start, (int, float)):
                        start = int(start)
                    if isinstance(end, (int, float)):
                        end = int(end)
                    idx_tuple.append(slice(start, end))
                else:
                    val = self._eval_ast(idx_spec)
                    if isinstance(val, (int, float)):
                        val = int(val)
                    idx_tuple.append(val)

            return var[tuple(idx_tuple)]

        if node_type == 'func_with_axis':
            func_name = ast[1]
            arg_ast = ast[2]
            axis = ast[3]  # Can be None or an int

            arg = self._eval_ast(arg_ast)

            if axis is not None:
                if isinstance(axis, tuple) and axis[0] == 'num':
                    axis = int(axis[1]) if isinstance(axis[1], (int, float)) else axis[1]
                elif isinstance(axis, (int, float)):
                    axis = int(axis)
                elif isinstance(axis, str) and axis.lstrip('-').isdigit():
                    axis = int(axis)

            if func_name == 'mean':
                return jnp.mean(arg, axis=axis) if axis is not None else jnp.mean(arg)
            if func_name == 'var':
                return jnp.var(arg, axis=axis) if axis is not None else jnp.var(arg)
            if func_name == 'std':
                return jnp.std(arg, axis=axis) if axis is not None else jnp.std(arg)
            if func_name == 'norm':
                return jnp.linalg.norm(arg, axis=axis) if axis is not None else jnp.linalg.norm(arg)
            if func_name == 'sum':
                return jnp.sum(arg, axis=axis) if axis is not None else jnp.sum(arg)
            if func_name == 'max':
                return jnp.max(arg, axis=axis) if axis is not None else jnp.max(arg)
            if func_name == 'min':
                return jnp.min(arg, axis=axis) if axis is not None else jnp.min(arg)
            if func_name == 'sqrt':
                return jnp.sqrt(arg)
            if func_name == 'log':
                return jnp.log(arg)
            if func_name == 'exp':
                return jnp.exp(arg)
            if func_name == 'softmax':
                axis = axis if axis is not None else -1
                return jax.nn.softmax(arg, axis=axis)
            if func_name == 'abs':
                return jnp.abs(arg)

        if node_type == 'topk':
            arg = self._eval_ast(ast[1])
            k = self._eval_ast(ast[2])
            if isinstance(k, (int, float)):
                k = int(k)
            # Return top-k values (not indices)
            flat = arg.flatten()
            top_vals = jnp.sort(flat)[-k:]
            return top_vals

        if node_type == 'threshold':
            arg = self._eval_ast(ast[1])
            t = self._eval_ast(ast[2])
            return (arg > t).astype(jnp.float32)

        if node_type == 'where':
            arg = self._eval_ast(ast[1])
            return jnp.where(arg)

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
            if l.strip() and not l.strip().startswith('\u235d') and not l.strip().startswith('//')
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
