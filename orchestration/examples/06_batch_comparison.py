"""Compare multiple pruning methods at once."""

import numpy as np
from apl_pruning import MiniAPLParser
import time

# Simulate a real layer (LLaMA-like dimensions)
W = np.random.randn(4096, 4096).astype(np.float32)
act = np.random.randn(512, 4096).astype(np.float32)
grad = np.random.randn(4096, 4096).astype(np.float32)

parser = MiniAPLParser()
parser.set_variables(W=W, act=act, grad=grad)

methods = {
    "Magnitude": "|W|",
    "Wanda": "|W| x mean(|act|)",
    "Grad x Weight": "|grad| x |W|",
    "Per-neuron norm": "norm(W, dim=-1)",
    "Per-neuron mean": "mean(|W|, dim=-1)",
    "Per-neuron std": "std(W, dim=-1)",
    "Combined": "|W| x mean(|W|, dim=-1)",
}

print(f"{'Method':<25} {'Time (ms)':<12} {'Shape':<20} {'Min':<10} {'Max':<10} {'Mean':<10}")
print("-" * 87)

for name, formula in methods.items():
    t0 = time.time()
    scores = parser.evaluate(formula)
    elapsed = (time.time() - t0) * 1000
    
    shape_str = str(scores.shape) if hasattr(scores, 'shape') else 'scalar'
    if hasattr(scores, 'shape') and scores.ndim > 0:
        min_v, max_v, mean_v = scores.min(), scores.max(), scores.mean()
    else:
        min_v = max_v = mean_v = float(scores) if not hasattr(scores, 'shape') else scores
    
    print(f"{name:<25} {elapsed:<12.2f} {shape_str:<20} {min_v:<10.4f} {max_v:<10.4f} {mean_v:<10.4f}")
