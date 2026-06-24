"""Layer cache: avoids recomputing common sub-expressions across layers."""

import numpy as np


class LayerCache:
    """Cache intermediate results per layer to avoid recomputation.
    
    Example:
        cache = LayerCache()
        cache.store("L0", "W_abs", np.abs(W0))
        cache.store("L0", "act_mean", np.mean(np.abs(act0)))
        # ...
        w_abs = cache.get("L0", "W_abs")  # cached, no recompute
    """
    
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0
    
    def store(self, layer_name: str, key: str, value: np.ndarray):
        """Store a computed value for a layer."""
        if layer_name not in self._cache:
            self._cache[layer_name] = {}
        self._cache[layer_name][key] = value
    
    def get(self, layer_name: str, key: str):
        """Retrieve a cached value. Returns None if not found."""
        if layer_name in self._cache and key in self._cache[layer_name]:
            self._hits += 1
            return self._cache[layer_name][key]
        self._misses += 1
        return None
    
    def has(self, layer_name: str, key: str) -> bool:
        """Check if a value is cached."""
        return layer_name in self._cache and key in self._cache[layer_name]
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def stats(self):
        return {"hits": self._hits, "misses": self._misses, 
                "total": self._hits + self._misses}


def fused_abs_mean(x, axis=None):
    """Compute mean(|x|) in one pass to avoid storing |x| intermediate."""
    return np.mean(np.abs(x), axis=axis)


def fused_abs_max(x, axis=None):
    """Compute max(|x|) in one pass."""
    return np.max(np.abs(x), axis=axis)


def fused_abs_sum(x, axis=None):
    """Compute sum(|x|) in one pass."""
    return np.sum(np.abs(x), axis=axis)


def fused_norm_ratio(a, b):
    """Compute norm(a) / norm(b) efficiently."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-12:
        raise ZeroDivisionError("Division by near-zero norm")
    return norm_a / norm_b
