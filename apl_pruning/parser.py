"""
Mini APL parser for pruning formulas - Level 2.
Translates a minimal APL subset into executable numpy code.

New in v0.2.0:
    - Axis support: mean(X, dim=-1), sum(X, dim=0), etc.
    - Indexing: W[i], W[:, j], W[0:512]
    - New primitives: std, softmax, topk, threshold, where
    - Broadcasting-aware operations with auto-reshape
    - Better error messages with shape info
    - Safe operations: no NaN from empty tensors, no silent div-by-zero
"""

import numpy as jnp
from typing import Dict, Any, Tuple


def _softmax_np(x, axis=-1):
    """Softmax in pure numpy."""
    e_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)


def _safe_mean(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute mean of empty tensor")
    return jnp.mean(x, axis=axis)


def _safe_var(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute variance of empty tensor")
    return jnp.var(x, axis=axis)


def _safe_std(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute std of empty tensor")
    return jnp.std(x, axis=axis)


def _safe_sum(x, axis=None):
    if x.size == 0:
        return jnp.array(0.0)
    return jnp.sum(x, axis=axis)


def _safe_norm(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute norm of empty tensor")
    return jnp.linalg.norm(x, axis=axis)


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
        X / Y         : division (raises on div-by-zero)
        X ^ Y         : power

        X[i]          : index row i
        X[i, j]       : index element (i,j)
        X[:, j]       : slice column j
        X[start:end]  : slice rows from start to end

        scores <- expr : assignment (use <- not the unicode arrow)

    Axis syntax:
        mean(X, dim=-1)   : mean along last axis
        sum(X, dim=0)     : sum along first axis
        norm(X, dim=-1)   : norm along last axis (per-vector)
    """

    def __init__(self):
        self.variables: Dict[str, Any] = {}

    def set_variable(self, name: str, value):
        self.variables[name] = value

    def set_variables(self, **kwargs):
        self.variables.update(kwargs)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _tokenize(self, code: str) -> list:
        tokens = []
        i = 0
        while i < len(code):
            c = code[i]

            if c.isspace():
                i += 1
                continue

            if c == '\u235d':
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue
            if code[i:i+2] == '//':
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue

            if code[i:i+2] == '+/':
                tokens.append(('REDUCE_SUM', '+/'))
                i += 2
                continue
            if code[i:i+2] == '<-':
                tokens.append(('ASSIGN', '<-'))
                i += 2
                continue

            if c == ':':
                tokens.append(('COLON', c))
                i += 1
                continue
            if c == ',':
                tokens.append(('COMMA', c))
                i += 1
                continue
            if c == '=':
                tokens.append(('EQUALS', c))
                i += 1
                continue

            if c in '+x/^':
                tokens.append(('OP', c))
                i += 1
                continue
            if c == '-':
                prev_types = [t[0] for t in tokens]
                if not tokens or prev_types[-1] in ('OP', 'LPAREN', 'ASSIGN', 'COMMA', 'LBRACKET', 'EQUALS', 'COLON'):
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
                tokens.append(('ASSIGN', '<-'))
                i += 1
                continue

            if c.isalpha() or c == '_':
                start = i
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    i += 1
                word = code[start:i]
                if word in ('mean', 'var', 'norm', 'sum', 'max', 'min',
                            'sqrt', 'log', 'exp', 'abs', 'std',
                            'softmax', 'topk', 'threshold', 'where'):
                    tokens.append(('FUNC', word))
                elif word in ('dim', 'axis'):
                    tokens.append(('KWARG', word))
                else:
                    tokens.append(('VAR', word))
                continue

            if c.isdigit() or (c == '.' and i+1 < len(code) and code[i+1].isdigit()):
                start = i
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
        parse_expression_ref = [None]

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
                ast, new_pos = parse_expression_ref[0](tokens, pos)
                pos = new_pos
                if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                    pos += 1
                return ast

            if tok_type == 'FUNC':
                func_name = tok_val
                pos += 1

                if pos < len(tokens) and tokens[pos][0] == 'LPAREN':
                    pos += 1
                    args = []
                    kwargs = {}

                    while pos < len(tokens) and tokens[pos][0] != 'RPAREN':
                        if tokens[pos][0] == 'COMMA':
                            pos += 1
                            continue

                        if (pos + 2 < len(tokens) and
                            tokens[pos][0] == 'KWARG' and
                            tokens[pos+1][0] == 'EQUALS'):
                            kw_name = tokens[pos][1]
                            pos += 2
                            kw_ast, new_pos = parse_expression_ref[0](tokens, pos)
                            pos = new_pos
                            kwargs[kw_name] = kw_ast
                        else:
                            arg_ast, new_pos = parse_expression_ref[0](tokens, pos)
                            pos = new_pos
                            if arg_ast is not None:
                                args.append(arg_ast)

                        if pos < len(tokens) and tokens[pos][0] == 'COMMA':
                            pos += 1

                    if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                        pos += 1

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
                        axis = kwargs.get('dim', kwargs.get('axis', None))
                        if axis is not None:
                            if isinstance(axis, tuple) and axis[0] == 'num':
                                axis = axis[1]
                            elif isinstance(axis, tuple) and axis[0] == 'neg':
                                inner = axis[1]
                                if isinstance(inner, tuple) and inner[0] == 'num':
                                    axis = -inner[1]
                        return ('func_with_axis', func_name, args[0] if args else None, axis)
                else:
                    arg = parse_primary()
                    return ('func_with_axis', func_name, arg, None)

            if tok_type == 'REDUCE_SUM':
                pos += 1
                arg = parse_primary()
                return ('func_with_axis', 'sum', arg, None)

            if tok_type == 'VAR':
                var_name = tok_val
                pos += 1

                if pos < len(tokens) and tokens[pos][0] == 'LBRACKET':
                    pos += 1
                    indices = []

                    while pos < len(tokens) and tokens[pos][0] != 'RBRACKET':
                        if tokens[pos][0] == 'COMMA':
                            pos += 1
                            continue

                        if tokens[pos][0] == 'COLON':
                            start_idx = ('num', 0)
                            pos += 1
                            if pos < len(tokens) and tokens[pos][0] not in ('RBRACKET', 'COMMA'):
                                end_val = parse_primary()
                            else:
                                end_val = ('num', None)
                            indices.append(('slice', start_idx, end_val))
                            continue

                        idx_val = parse_primary()
                        if pos < len(tokens) and tokens[pos][0] == 'COLON':
                            pos += 1
                            if pos < len(tokens) and tokens[pos][0] not in ('RBRACKET', 'COMMA'):
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

                    if len(indices) == 1 and not (isinstance(indices[0], tuple) and indices[0][0] == 'slice'):
                        return ('index', var_name, indices[0])
                    else:
                        return ('index_multi', var_name, indices)

                return ('var', var_name)

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

        parse_expression_ref[0] = self._parse_expression

        ast = parse_primary()
        if ast is not None:
            ast = parse_binary(ast)
        return (ast, pos)

    def parse_expression(self, code: str):
        tokens = self._tokenize(code)
        ast, _ = self._parse_expression(tokens, 0)
        return ast

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------

    def _broadcast_mul(self, a, b):
        if a.ndim == 1 and b.ndim == 2 and a.shape[0] == b.shape[0]:
            a = a.reshape(-1, 1)
        elif b.ndim == 1 and a.ndim == 2 and b.shape[0] == a.shape[0]:
            b = b.reshape(-1, 1)
        return a * b

    def _eval_ast(self, ast) -> Any:
        if ast is None:
            return None

        node_type = ast[0]

        # --- Literals ---
        if node_type == 'num':
            return ast[1]

        if node_type == 'var':
            var_name = ast[1]
            if var_name not in self.variables:
                raise NameError(
                    f"Variable '{var_name}' not defined. "
                    f"Available variables: {list(self.variables.keys())}"
                )
            return self.variables[var_name]

        # --- Unary ---
        if node_type == 'neg':
            return -self._eval_ast(ast[1])

        if node_type == 'abs':
            return jnp.abs(self._eval_ast(ast[1]))

        # --- Indexing ---
        if node_type == 'index':
            var_name = ast[1]
            idx_ast = ast[2]
            var = self.variables[var_name]
            idx = self._eval_ast(idx_ast)
            if isinstance(idx, (int, float, jnp.integer, jnp.floating)):
                idx = int(idx)
            return var[idx]

        if node_type == 'index_multi':
            var_name = ast[1]
            indices = ast[2]
            var = self.variables[var_name]
            idx_tuple = []
            for idx_spec in indices:
                if isinstance(idx_spec, tuple) and idx_spec[0] == 'slice':
                    start = self._eval_ast(idx_spec[1]) if idx_spec[1] else 0
                    end = self._eval_ast(idx_spec[2]) if idx_spec[2] else None
                    if isinstance(start, (int, float, jnp.integer, jnp.floating)):
                        start = int(start)
                    if isinstance(end, (int, float, jnp.integer, jnp.floating)):
                        end = int(end)
                    idx_tuple.append(slice(start, end))
                else:
                    val = self._eval_ast(idx_spec)
                    if isinstance(val, (int, float, jnp.integer, jnp.floating)):
                        val = int(val)
                    idx_tuple.append(val)
            return var[tuple(idx_tuple)]

        # --- Functions with axis ---
        if node_type == 'func_with_axis':
            func_name = ast[1]
            arg_ast = ast[2]
            axis = ast[3]

            arg = self._eval_ast(arg_ast)

            # Normalize axis
            if axis is not None:
                if isinstance(axis, tuple) and axis[0] == 'num':
                    axis = int(axis[1]) if isinstance(axis[1], (int, float)) else axis[1]
                elif isinstance(axis, tuple) and axis[0] == 'neg':
                    inner = axis[1]
                    if isinstance(inner, tuple) and inner[0] == 'num':
                        axis = -int(inner[1]) if isinstance(inner[1], (int, float)) else -inner[1]
                elif isinstance(axis, (int, float, jnp.integer, jnp.floating)):
                    axis = int(axis)

            # --- Safe reductions ---
            if func_name == 'mean':
                return _safe_mean(arg, axis=axis) if axis is not None else _safe_mean(arg)
            if func_name == 'var':
                return _safe_var(arg, axis=axis) if axis is not None else _safe_var(arg)
            if func_name == 'std':
                return _safe_std(arg, axis=axis) if axis is not None else _safe_std(arg)
            if func_name == 'norm':
                return _safe_norm(arg, axis=axis) if axis is not None else _safe_norm(arg)
            if func_name == 'sum':
                return _safe_sum(arg, axis=axis) if axis is not None else _safe_sum(arg)

            # --- Min/max with empty check ---
            if func_name == 'max':
                if arg.size == 0:
                    raise ValueError("Cannot compute max of empty tensor")
                return jnp.max(arg, axis=axis) if axis is not None else jnp.max(arg)
            if func_name == 'min':
                if arg.size == 0:
                    raise ValueError("Cannot compute min of empty tensor")
                return jnp.min(arg, axis=axis) if axis is not None else jnp.min(arg)

            # --- Math with domain validation ---
            if func_name == 'sqrt':
                if jnp.any(arg < 0):
                    raise ValueError("Cannot compute sqrt of negative values. Use |X| first.")
                return jnp.sqrt(arg)
            if func_name == 'log':
                if jnp.any(arg <= 0):
                    raise ValueError("Cannot compute log of non-positive values. Use |X| or X + epsilon.")
                return jnp.log(arg)

            # --- Safe passthrough ---
            if func_name == 'exp':
                return jnp.exp(arg)
            if func_name == 'softmax':
                axis = axis if axis is not None else -1
                return _softmax_np(arg, axis=axis)
            if func_name == 'abs':
                return jnp.abs(arg)

            raise ValueError(f"Unknown function: {func_name}")

        # --- Special functions ---
        if node_type == 'topk':
            arg = self._eval_ast(ast[1])
            k = self._eval_ast(ast[2])
            if isinstance(k, (int, float, jnp.integer, jnp.floating)):
                k = int(k)
            flat = arg.flatten()
            return jnp.sort(flat)[-k:]

        if node_type == 'threshold':
            arg = self._eval_ast(ast[1])
            t = self._eval_ast(ast[2])
            return (arg > t).astype(jnp.float32)

        if node_type == 'where':
            arg = self._eval_ast(ast[1])
            return jnp.where(arg)

        # --- Binary operations ---
        if node_type == 'add':
            return self._eval_ast(ast[1]) + self._eval_ast(ast[2])
        if node_type == 'sub':
            return self._eval_ast(ast[1]) - self._eval_ast(ast[2])
        if node_type == 'mul':
            return self._broadcast_mul(self._eval_ast(ast[1]), self._eval_ast(ast[2]))
        if node_type == 'div':
            a = self._eval_ast(ast[1])
            b = self._eval_ast(ast[2])
            if jnp.any(jnp.abs(b) < 1e-12):
                raise ZeroDivisionError(
                    f"Division by near-zero. min(|denominator|) = {jnp.min(jnp.abs(b)):.2e}"
                )
            return a / b
        if node_type == 'pow':
            return self._eval_ast(ast[1]) ** self._eval_ast(ast[2])

        raise ValueError(f"Unknown AST node type: {node_type}")

    # ------------------------------------------------------------------
    # Compilation and execution
    # ------------------------------------------------------------------

    def compile(self, code: str):
        lines = [
            l.strip() for l in code.strip().split('\n')
            if l.strip() and not l.strip().startswith('\u235d') and not l.strip().startswith('//')
        ]

        if len(lines) == 1 and '<-' not in lines[0] and '\u2190' not in lines[0]:
            ast = self.parse_expression(lines[0])
            def fn():
                return self._eval_ast(ast)
            return fn

        final_expr = None
        for line in lines:
            is_assignment = False
            var_name = None
            expr = None
            if '<-' in line:
                parts = line.split('<-', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()
                is_assignment = True
            elif '\u2190' in line:
                parts = line.split('\u2190', 1)
                var_name = parts[0].strip()
                expr = parts[1].strip()
                is_assignment = True
            if is_assignment:
                ast = self.parse_expression(expr)
                result = self._eval_ast(ast)
                self.variables[var_name] = result
            else:
                final_expr = self.parse_expression(line)

        def fn():
            return self._eval_ast(final_expr) if final_expr else None
        return fn

    def evaluate(self, code: str):
        fn = self.compile(code)
        return fn()

    def __call__(self, code: str):
        return self.evaluate(code)
