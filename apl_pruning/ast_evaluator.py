"""Evaluates an AST into numpy arrays."""

import numpy as np
from apl_pruning.utils import (
    softmax, broadcast_add, broadcast_sub, broadcast_mul, broadcast_div,
    safe_mean, safe_var, safe_std, safe_sum, safe_norm
)


def eval_ast(ast, variables: dict):
    """Evaluate an AST node. `variables` is a dict mapping names to arrays."""
    
    if ast is None:
        return None
    
    node_type = ast[0]
    
    # ---- Literals ----
    if node_type == 'num':
        return ast[1]
    
    if node_type == 'var':
        name = ast[1]
        if name not in variables:
            raise NameError(
                f"Variable '{name}' not defined. "
                f"Available: {list(variables.keys())}"
            )
        return variables[name]
    
    # ---- Unary ----
    if node_type == 'neg':
        return -eval_ast(ast[1], variables)
    
    if node_type == 'abs':
        return np.abs(eval_ast(ast[1], variables))
    
    # ---- Indexing ----
    if node_type == 'index':
        name, idx_ast = ast[1], ast[2]
        var = variables[name]
        idx = _to_int(eval_ast(idx_ast, variables))
        return var[idx]
    
    if node_type == 'index_multi':
        name, indices = ast[1], ast[2]
        var = variables[name]
        idx_tuple = _build_index_tuple(indices, variables)
        return var[tuple(idx_tuple)]
    
    # ---- Functions ----
    if node_type == 'func_with_axis':
        return _eval_function(ast, variables)
    
    # ---- Special forms ----
    if node_type == 'topk':
        arg = eval_ast(ast[1], variables)
        k = _to_int(eval_ast(ast[2], variables))
        return np.sort(arg.flatten())[-k:]
    
    if node_type == 'threshold':
        arg = eval_ast(ast[1], variables)
        t = eval_ast(ast[2], variables)
        return (arg > t).astype(np.float32)
    
    if node_type == 'where':
        arg = eval_ast(ast[1], variables)
        return np.where(arg)
    
    # ---- Binary ----
    if node_type == 'add':
        return broadcast_add(eval_ast(ast[1], variables), eval_ast(ast[2], variables))
    if node_type == 'sub':
        return broadcast_sub(eval_ast(ast[1], variables), eval_ast(ast[2], variables))
    if node_type == 'mul':
        return broadcast_mul(eval_ast(ast[1], variables), eval_ast(ast[2], variables))
    if node_type == 'div':
        return broadcast_div(eval_ast(ast[1], variables), eval_ast(ast[2], variables))
    if node_type == 'pow':
        return eval_ast(ast[1], variables) ** eval_ast(ast[2], variables)
    
    raise ValueError(f"Unknown AST node type: {node_type}")


# ---- Internal helpers ----

def _to_int(val):
    """Safely convert a value to int."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    return val


def _build_index_tuple(indices, variables):
    """Build a tuple for numpy advanced indexing."""
    result = []
    for spec in indices:
        if isinstance(spec, tuple) and spec[0] == 'slice':
            start = eval_ast(spec[1], variables) if spec[1] else 0
            end = eval_ast(spec[2], variables) if spec[2] else None
            result.append(slice(_to_int(start), _to_int(end)))
        else:
            result.append(_to_int(eval_ast(spec, variables)))
    return result


def _normalize_axis(axis):
    """Convert axis from AST representation to int or None."""
    if axis is None:
        return None
    if isinstance(axis, tuple):
        if axis[0] == 'num':
            return int(axis[1]) if isinstance(axis[1], (int, float)) else axis[1]
        if axis[0] == 'neg':
            inner = axis[1]
            if isinstance(inner, tuple) and inner[0] == 'num':
                return -int(inner[1]) if isinstance(inner[1], (int, float)) else -inner[1]
    if isinstance(axis, (int, float, np.integer, np.floating)):
        return int(axis)
    return axis


def _eval_function(ast, variables):
    """Evaluate a func_with_axis node."""
    func_name = ast[1]
    arg = eval_ast(ast[2], variables)
    axis = _normalize_axis(ast[3])
    
    # Safe reductions
    if func_name == 'mean':
        return safe_mean(arg, axis=axis) if axis is not None else safe_mean(arg)
    if func_name == 'var':
        return safe_var(arg, axis=axis) if axis is not None else safe_var(arg)
    if func_name == 'std':
        return safe_std(arg, axis=axis) if axis is not None else safe_std(arg)
    if func_name == 'norm':
        return safe_norm(arg, axis=axis) if axis is not None else safe_norm(arg)
    if func_name == 'sum':
        return safe_sum(arg, axis=axis) if axis is not None else safe_sum(arg)
    
    # Min/max with empty check
    if func_name == 'max':
        if arg.size == 0:
            raise ValueError("Cannot compute max of empty tensor")
        return np.max(arg, axis=axis) if axis is not None else np.max(arg)
    if func_name == 'min':
        if arg.size == 0:
            raise ValueError("Cannot compute min of empty tensor")
        return np.min(arg, axis=axis) if axis is not None else np.min(arg)
    
    # Math with domain validation
    if func_name == 'sqrt':
        if np.any(arg < 0):
            raise ValueError("Cannot compute sqrt of negative values. Use |X| first.")
        return np.sqrt(arg)
    if func_name == 'log':
        if np.any(arg <= 0):
            raise ValueError("Cannot compute log of non-positive values. Use |X| or X + epsilon.")
        return np.log(arg)
    
    # Safe passthrough
    if func_name == 'exp':
        return np.exp(arg)
    if func_name == 'softmax':
        return softmax(arg, axis=axis if axis is not None else -1)
    if func_name == 'abs':
        return np.abs(arg)
    
    raise ValueError(f"Unknown function: {func_name}")
