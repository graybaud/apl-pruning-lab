"""Per-neuron scoring with axis support."""

import numpy as np
from apl_pruning import MiniAPLParser

# Use consistent dimensions: W(rows, cols), act(batch, rows)
W = np.random.randn(768, 3072).astype(np.float32)
act = np.random.randn(128, 768).astype(np.float32)

parser = MiniAPLParser()
parser.set_variables(W=W, act=act)

# mean(|act|) is scalar, broadcasts with W(768,3072) -> (768,3072)
scores_scalar = parser.evaluate("|W| x mean(|act|)")
print(f"Wanda (scalar mean): shape {scores_scalar.shape}")

# Per-neuron norm: norm along last axis -> (768,)
norms = parser.evaluate("norm(W, dim=-1)")
print(f"Per-neuron norms: shape {norms.shape}")
print(f"First 5 norms: {norms[:5]}")

# Per-neuron mean weight: mean along last axis -> (768,)
mean_w = parser.evaluate("mean(|W|, dim=-1)")
print(f"Per-neuron mean |W|: shape {mean_w.shape}")
print(f"First 5 means: {mean_w[:5]}")

# Combine: |W| (768,3072) * mean(|W|, dim=-1) (768,) broadcasts -> (768,3072)
per_neuron = parser.evaluate("|W| x mean(|W|, dim=-1)")
print(f"Per-neuron weighted: shape {per_neuron.shape}")
print(f"First neuron first 3 weights: {per_neuron[0, :3]}")
