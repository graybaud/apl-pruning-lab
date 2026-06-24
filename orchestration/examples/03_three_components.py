"""Three-component pruning score: direction x selectivity x distortion."""

import numpy as np
from apl_pruning import MiniAPLParser

# Simulate a layer
W = np.random.randn(768, 3072).astype(np.float32)
act = np.random.randn(128, 768).astype(np.float32)

# Similarity matrices (simplified: random symmetric positive definite)
S_full = np.eye(128)
noise = np.random.randn(128, 128) * 0.1
S_full = S_full + noise @ noise.T
S_without = S_full.copy()
S_without[0, :] *= 0.5
S_without[:, 0] *= 0.5

parser = MiniAPLParser()
parser.set_variables(W=W, act=act, S_full=S_full, S_without=S_without)

score = parser.evaluate("""
    direction <- (max(|W|)) / mean(|W|)
    selectivity <- var(act) / mean(act)
    distortion <- norm(S_full - S_without) / norm(S_full)
    direction x selectivity x distortion
""")

print(f"Direction:  {parser.variables['direction']:.4f}")
print(f"Selectivity: {parser.variables['selectivity']:.4f}")
print(f"Distortion: {parser.variables['distortion']:.4f}")
print(f"Final score: {score:.4f}")

# Verify
d = np.max(np.abs(W)) / np.mean(np.abs(W))
s = np.var(act) / np.mean(act)
dist = np.linalg.norm(S_full - S_without) / np.linalg.norm(S_full)
expected = d * s * dist
print(f"Matches numpy: {np.allclose(score, expected)}")
