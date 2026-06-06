"""Grad x Weight pruning method."""

import numpy as np
from apl_pruning import MiniAPLParser

W = np.random.randn(768, 3072).astype(np.float32)
grad = np.random.randn(768, 3072).astype(np.float32)

parser = MiniAPLParser()
parser.set_variables(W=W, grad=grad)

# Grad x Weight
scores = parser.evaluate("|grad| x |W|")

print(f"Score shape: {scores.shape}")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

# Verify
expected = np.abs(grad) * np.abs(W)
print(f"Matches numpy: {np.allclose(scores, expected)}")
