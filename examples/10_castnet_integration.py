"""Example: using apl-pruning as a drop-in replacement for CastNet scorers.

This shows how the APL formulas produce identical results to the original
hardcoded PyTorch implementations.
"""

import numpy as np
import torch
from apl_pruning.castnet_bridge import compare_all_methods
from apl_pruning.scorers import METHODS, list_methods, get_pytorch

print("=" * 60)
print("  APL-PRUNING × CASTNET INTEGRATION")
print("=" * 60)

# Simulate a layer (like OPT/LLaMA fc1: 768 -> 3072)
W = torch.randn(768, 3072)
act = torch.randn(128, 768)
grad = torch.randn(768, 3072)

print(f"\n  Layer shape: W={list(W.shape)}, act={list(act.shape)}, grad={list(grad.shape)}")

# Compare all applicable methods
print("\n  --- Method Comparison ---")
results = compare_all_methods(W, act=act, grad=grad)
for method, scores in results.items():
    desc = METHODS[method]["description"]
    print(f"  {method:<22} mean={scores.mean():.6f}  |  {desc}")

# Show PyTorch export for each method
print("\n  --- PyTorch Export (for verification) ---")
for method in ["wanda", "gradient", "magnitude", "direction"]:
    code = get_pytorch(method)
    print(f"  {method}:")
    print(f"    {code}")

# Show all available methods
print("\n  --- All Available Methods ---")
for m in list_methods():
    print(f"  {m['name']:<22} needs_grad={m['needs_grad']:<6} {m['description']}")

print("\n" + "=" * 60)
print("  Integration complete. Use these in CastNet directly.")
print("=" * 60)
