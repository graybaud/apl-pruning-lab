"""Score pruning importance across all layers of a model."""

import numpy as np
from apl_pruning import LayerScorer

# Simulate a 4-layer model
layers = {}
for i in range(4):
    layers[f"layer_{i}"] = {
        "W": np.random.randn(768, 3072).astype(np.float32),
        "act": np.random.randn(128, 768).astype(np.float32),
    }

scorer = LayerScorer(layers)

# Score all layers with Wanda
print("=== Wanda across all layers ===")
scores = scorer.score_all("|W| x mean(|act|)")
for name, score in scores.items():
    print(f"{name}: shape={score.shape}, mean={score.mean():.4f}, max={score.max():.4f}")

# Compare multiple methods
print("\n=== Comparing methods ===")
methods = {
    "Magnitude": "|W|",
    "Wanda": "|W| x mean(|act|)",
    "Per-neuron norm": "norm(W, dim=-1)",
}
results = scorer.compare(methods)
scorer.summary(results)
