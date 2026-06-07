"""Utility functions for APL pruning DSL."""

import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _ensure_array(x):
    """Convert Python scalars to numpy arrays."""
    if isinstance(x, (int, float)):
        return np.array(x)
    return x


def _broadcast_for_binary(a, b):
    """Prepare two arrays for binary operation with APL-style broadcasting."""
    a = _ensure_array(a)
    b = _ensure_array(b)
    
    if a.ndim == 1 and b.ndim >= 2 and a.shape[0] == b.shape[0]:
        a = a.reshape(-1, *([1] * (b.ndim - 1)))
    elif b.ndim == 1 and a.ndim >= 2 and b.shape[0] == a.shape[0]:
        b = b.reshape(-1, *([1] * (a.ndim - 1)))
    
    return a, b


def broadcast_add(a, b):
    a, b = _broadcast_for_binary(a, b)
    try:
        return a + b
    except ValueError as e:
        raise ValueError(f"Dimension mismatch: {a.shape} + {b.shape} cannot be broadcast. {e}")


def broadcast_sub(a, b):
    a, b = _broadcast_for_binary(a, b)
    try:
        return a - b
    except ValueError as e:
        raise ValueError(f"Dimension mismatch: {a.shape} - {b.shape} cannot be broadcast. {e}")


def broadcast_mul(a, b):
    a, b = _broadcast_for_binary(a, b)
    try:
        return a * b
    except ValueError as e:
        raise ValueError(f"Dimension mismatch: {a.shape} x {b.shape} cannot be broadcast. {e}")


def broadcast_div(a, b):
    a, b = _broadcast_for_binary(a, b)
    if np.any(np.abs(b) < 1e-12):
        raise ZeroDivisionError(
            f"Division by near-zero. min(|denominator|) = {np.min(np.abs(b)):.2e}"
        )
    try:
        return a / b
    except ValueError as e:
        raise ValueError(f"Dimension mismatch: {a.shape} / {b.shape} cannot be broadcast. {e}")


# ---- Safe math functions ----

def safe_mean(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute mean of empty tensor")
    return np.mean(x, axis=axis)


def safe_var(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute variance of empty tensor")
    return np.var(x, axis=axis)


def safe_std(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute std of empty tensor")
    return np.std(x, axis=axis)


def safe_sum(x, axis=None):
    if x.size == 0:
        return np.array(0.0)
    return np.sum(x, axis=axis)


def safe_norm(x, axis=None):
    if x.size == 0:
        raise ValueError("Cannot compute norm of empty tensor")
    return np.linalg.norm(x, axis=axis)


def safe_rank(x):
    """Matrix rank via SVD thresholding."""
    if x.size == 0:
        raise ValueError("Cannot compute rank of empty tensor")
    if x.ndim < 2:
        return 1 if x.size > 0 else 0
    s = np.linalg.svd(x, compute_uv=False)
    tol = s.max() * max(x.shape) * np.finfo(s.dtype).eps
    return int(np.sum(s > tol))


def safe_sort(x, axis=None):
    """Sort that raises on empty tensor."""
    if x.size == 0:
        raise ValueError("Cannot sort empty tensor")
    return np.sort(x, axis=axis)
