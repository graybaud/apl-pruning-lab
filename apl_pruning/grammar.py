"""Grammar: parses tokens into an Abstract Syntax Tree."""

from typing import Any, Tuple


def parse_expression(tokens: list, pos: int = 0) -> Tuple[Any, int]:
    """Parse tokens into an AST. Returns (ast, new_position)."""
    
    parse_ref = [None]  # Forward reference for recursion
    
    def parse_primary():
        nonlocal pos
        if pos >= len(tokens):
            return None
        
        tok_type, tok_val = tokens[pos]
        
        # Negation
        if tok_type == 'NEG':
            pos += 1
            return ('neg', parse_primary())
        
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
            ast, new_pos = parse_ref[0](tokens, pos)
            pos = new_pos
            if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
                pos += 1
            return ast
        
        # Function calls
        if tok_type == 'FUNC':
            return _parse_function()
        
        # Reduction
        if tok_type == 'REDUCE_SUM':
            pos += 1
            return ('func_with_axis', 'sum', parse_primary(), None)
        
        # Variables with optional indexing
        if tok_type == 'VAR':
            return _parse_variable()
        
        # Numbers
        if tok_type == 'NUM':
            pos += 1
            return ('num', tok_val)
        
        pos += 1
        return None
    
    def _parse_function():
        nonlocal pos
        func_name = tokens[pos][1]
        pos += 1
        
        # No parentheses: func arg
        if pos >= len(tokens) or tokens[pos][0] != 'LPAREN':
            return ('func_with_axis', func_name, parse_primary(), None)
        
        # Parenthesized: func(arg1, kwarg=val, ...)
        pos += 1  # skip (
        args = []
        kwargs = {}
        
        while pos < len(tokens) and tokens[pos][0] != 'RPAREN':
            if tokens[pos][0] == 'COMMA':
                pos += 1
                continue
            
            # Keyword argument: dim=0
            if (pos + 2 < len(tokens) and
                tokens[pos][0] == 'KWARG' and
                tokens[pos+1][0] == 'EQUALS'):
                kw_name = tokens[pos][1]
                pos += 2
                kw_ast, new_pos = parse_ref[0](tokens, pos)
                pos = new_pos
                kwargs[kw_name] = kw_ast
            else:
                arg_ast, new_pos = parse_ref[0](tokens, pos)
                pos = new_pos
                if arg_ast is not None:
                    args.append(arg_ast)
            
            if pos < len(tokens) and tokens[pos][0] == 'COMMA':
                pos += 1
        
        if pos < len(tokens) and tokens[pos][0] == 'RPAREN':
            pos += 1
        
        # Special functions
        if func_name == 'topk':
            k = args[1] if len(args) >= 2 else None
            return ('topk', args[0] if args else None, k)
        if func_name == 'threshold':
            t = args[1] if len(args) >= 2 else None
            return ('threshold', args[0] if args else None, t)
        if func_name == 'where':
            return ('where', args[0] if args else None)
        
        # Regular functions with optional axis
        axis = kwargs.get('dim', kwargs.get('axis', None))
        if axis is not None:
            if isinstance(axis, tuple):
                if axis[0] == 'num':
                    axis = axis[1]
                elif axis[0] == 'neg' and isinstance(axis[1], tuple) and axis[1][0] == 'num':
                    axis = -axis[1][1]
        
        return ('func_with_axis', func_name, args[0] if args else None, axis)
    
    def _parse_variable():
        nonlocal pos
        var_name = tokens[pos][1]
        pos += 1
        
        # No indexing
        if pos >= len(tokens) or tokens[pos][0] != 'LBRACKET':
            return ('var', var_name)
        
        # Indexing: X[i, j, start:end, ...]
        pos += 1  # skip [
        indices = []
        
        while pos < len(tokens) and tokens[pos][0] != 'RBRACKET':
            if tokens[pos][0] == 'COMMA':
                pos += 1
                continue
            
            # Bare colon: X[:, ...]
            if tokens[pos][0] == 'COLON':
                pos += 1
                if pos < len(tokens) and tokens[pos][0] not in ('RBRACKET', 'COMMA'):
                    end_val = parse_primary()
                else:
                    end_val = ('num', None)
                indices.append(('slice', ('num', 0), end_val))
                continue
            
            # Index or slice
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
        
        # Single non-slice index
        if len(indices) == 1 and not (
            isinstance(indices[0], tuple) and indices[0][0] == 'slice'
        ):
            return ('index', var_name, indices[0])
        
        return ('index_multi', var_name, indices)
    
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
    
    # Wire up forward reference
    parse_ref[0] = parse_expression
    
    ast = parse_primary()
    if ast is not None:
        ast = parse_binary(ast)
    
    return (ast, pos)
