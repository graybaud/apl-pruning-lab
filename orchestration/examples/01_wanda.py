"""Wanda pruning method in 1 line of APL."""

import numpy as np
from apl_pruning import MiniAPLParser

# Simulate a small layer
W = np.random.randn(768, 3072).astype(np.float32)
act = np.random.randn(128, 768).astype(np.float32)

parser = MiniAPLParser()
parser.set_variables(W=W, act=act)

# Wanda: |W| * mean(|activations|)
scores = parser.evaluate("|W| x mean(|act|)")

print(f"Weight shape: {W.shape}")
print(f"Activations shape: {act.shape}")
print(f"Scores shape: {scores.shape}")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Mean score: {scores.mean():.4f}")

# Verify against numpy
expected = np.abs(W) * np.mean(np.abs(act))
print(f"Matches numpy: {np.allclose(scores, expected)}")
