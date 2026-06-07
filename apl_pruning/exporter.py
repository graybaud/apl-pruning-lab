"""Export APL expressions to PyTorch code."""

import numpy as np

# Mapping APL functions to PyTorch
_UNARY_MAP = {
    'abs': 'torch.abs',
    'mean': 'torch.mean',
    'var': 'torch.var',
    'std': 'torch.std',
    'norm': 'torch.linalg.norm',
    'sum': 'torch.sum',
    'max': 'torch.max',
    'min': 'torch.min',
    'sqrt': 'torch.sqrt',
    'log': 'torch.log',
    'exp': 'torch.exp',
    'softmax': 'torch.nn.functional.softmax',
    'rank': 'torch.linalg.matrix_rank',
    'sort': 'torch.sort',
}

_BINARY_MAP = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
    'pow': '**',
}


def to_pytorch(ast_or_code, variables=None):
    """Convert an APL expression or AST to PyTorch code.
    
    Args:
        ast_or_code: Either an AST tuple (from parser.parse()) or APL code string.
        variables: Optional dict of variable names -> shapes for dim hints.
    
    Returns:
        str: PyTorch code string.
    
    Example:
        >>> from apl_pruning import MiniAPLParser
        >>> p = MiniAPLParser()
        >>> ast = p.parse("|W| x mean(|act|)")
        >>> print(to_pytorch(ast))
        torch.abs(W) * torch.mean(torch.abs(act))
    """
    from apl_pruning.parser import MiniAPLParser
    
    # If we got a string, parse it first
    if isinstance(ast_or_code, str):
        p = MiniAPLParser()
        # Set dummy variables so parsing doesn't fail on names
        if variables:
            p.set_variables(**variables)
        ast = p.parse(ast_or_code)
    else:
        ast = ast_or_code
    
    return _ast_to_pytorch(ast)


def _ast_to_pytorch(ast, indent=0):
    """Recursively convert AST to PyTorch code string."""
    if ast is None:
        return 'None'
    
    node_type = ast[0]
    
    # Literals
    if node_type == 'num':
        return str(ast[1])
    
    if node_type == 'var':
        return ast[1]
    
    # Unary
    if node_type == 'neg':
        return f"(-{_ast_to_pytorch(ast[1])})"
    
    if node_type == 'abs':
        return f"torch.abs({_ast_to_pytorch(ast[1])})"
    
    # Indexing
    if node_type == 'index':
        var = ast[1]
        idx = _ast_to_pytorch(ast[2])
        return f"{var}[{idx}]"
    
    if node_type == 'index_multi':
        var = ast[1]
        indices = []
        for spec in ast[2]:
            if isinstance(spec, tuple) and spec[0] == 'slice':
                start_ast = spec[1]
                end_ast = spec[2]
                # Check if this is a bare colon (X[:] or X[:, :])
                is_bare_start = (start_ast is None or 
                    (isinstance(start_ast, tuple) and start_ast[0] == 'num' and start_ast[1] == 0))
                is_bare_end = (end_ast is None or
                    (isinstance(end_ast, tuple) and end_ast[0] == 'num' and end_ast[1] is None))
                
                if is_bare_start and is_bare_end:
                    indices.append(':')
                elif is_bare_start:
                    indices.append(f":{_ast_to_pytorch(end_ast)}")
                elif is_bare_end:
                    indices.append(f"{_ast_to_pytorch(start_ast)}:")
                else:
                    indices.append(f"{_ast_to_pytorch(start_ast)}:{_ast_to_pytorch(end_ast)}")
            else:
                indices.append(_ast_to_pytorch(spec))
        return f"{var}[{', '.join(indices)}]"
    
    # Functions
    if node_type == 'func_with_axis':
        func_name = ast[1]
        arg = _ast_to_pytorch(ast[2])
        axis = ast[3]
        
        if func_name == 'threshold':
            t = _ast_to_pytorch(ast[2]) if len(ast) > 3 else '0.5'
            return f"({arg} > {t}).float()"
        
        if func_name == 'topk':
            k = _ast_to_pytorch(ast[2]) if len(ast) > 2 else '10'
            return f"torch.topk({arg}.flatten(), {k}).values"
        
        if func_name == 'where':
            return f"torch.where({arg})"
        
        torch_fn = _UNARY_MAP.get(func_name, func_name)
        
        if axis is not None:
            if isinstance(axis, tuple):
                if axis[0] == 'num':
                    axis = axis[1]
                elif axis[0] == 'neg' and isinstance(axis[1], tuple) and axis[1][0] == 'num':
                    axis = -axis[1][1]
            return f"{torch_fn}({arg}, dim={axis})"
        
        return f"{torch_fn}({arg})"
    
    # Special forms
    if node_type == 'topk':
        arg = _ast_to_pytorch(ast[1])
        k = _ast_to_pytorch(ast[2])
        return f"torch.topk({arg}.flatten(), {k}).values"
    
    if node_type == 'threshold':
        arg = _ast_to_pytorch(ast[1])
        t = _ast_to_pytorch(ast[2])
        return f"({arg} > {t}).float()"
    
    if node_type == 'where':
        arg = _ast_to_pytorch(ast[1])
        return f"torch.where({arg})"
    
    # Binary
    if node_type in _BINARY_MAP:
        left = _ast_to_pytorch(ast[1])
        right = _ast_to_pytorch(ast[2])
        op = _BINARY_MAP[node_type]
        return f"({left} {op} {right})"
    
    return f"<unknown:{node_type}>"


def to_pytorch_function(code, func_name=None, variables=None):
    """Export APL code as a complete Python function.
    
    Args:
        code: APL code string.
        func_name: Name for the generated function.
        variables: Dict of variable names -> shapes for type hints.
    
    Returns:
        str: Complete Python function with imports.
    
    Example:
        >>> print(to_pytorch_function("|W| x mean(|act|)", "wanda"))
        import torch
        def wanda(W, act):
            return torch.abs(W) * torch.mean(torch.abs(act))
    """
    from apl_pruning import MiniAPLParser
    
    p = MiniAPLParser()
    ast = p.parse(code)
    
    # Extract all variable names from AST
    var_names = _extract_variables(ast)
    
    # Build function
    if func_name is None:
        func_name = "pruning_score"
    
    body = _ast_to_pytorch(ast)
    
    lines = ["import torch", "import torch.nn.functional", ""]
    lines.append(f"def {func_name}({', '.join(sorted(var_names))}):")
    lines.append(f"    return {body}")
    
    return '\n'.join(lines)


def _extract_variables(ast):
    """Extract all variable names from an AST."""
    vars_found = set()
    _extract_vars_recursive(ast, vars_found)
    return vars_found


def _extract_vars_recursive(ast, vars_found):
    if ast is None:
        return
    if isinstance(ast, tuple):
        if ast[0] == 'var':
            vars_found.add(ast[1])
        elif ast[0] == 'index':
            vars_found.add(ast[1])
        elif ast[0] == 'index_multi':
            vars_found.add(ast[1])
        for item in ast[1:]:
            _extract_vars_recursive(item, vars_found)
