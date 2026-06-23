"""Compare pruning methods using the APL DSL."""

import numpy as np
from apl_pruning.scorers import METHODS, score_layer, compare_methods, list_methods, get_pytorch

# List all available methods
print("Available pruning methods:")
for m in list_methods():
    print(f"  {m['name']:<20} grad={m['needs_grad']:<6} {m['description']}")

# Simulate a layer
W = np.random.randn(768, 3072).astype(np.float32)
act = np.random.randn(128, 768).astype(np.float32)
grad = np.random.randn(768, 3072).astype(np.float32)

print("\n=== Individual methods ===")
for name in ["magnitude", "wanda", "gradient", "softmax_grad"]:
    if METHODS[name]["needs_grad"]:
        scores = score_layer(name, W=W, grad=grad, act=act)
    else:
        scores = score_layer(name, W=W, act=act)
    print(f"{name:<20} shape={scores.shape} min={scores.min():.4f} max={scores.max():.4f}")

# Compare multiple methods
print("\n=== Comparison ===")
results = compare_methods(
    ["magnitude", "wanda", "gradient", "direction", "norm_ratio"],
    W=W, act=act, grad=grad
)
for method, scores in results.items():
    print(f"{method:<20} mean={scores.mean():.4f}")

# Export to PyTorch
print("\n=== PyTorch Export ===")
print("Wanda:", get_pytorch("wanda"))
print("GPS:", get_pytorch("gps"))
