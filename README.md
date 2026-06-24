# APL Pruning Lab

Mini APL-like DSL for describing pruning formulas. Compiles to numpy.
55 built-in methods. Test 10 pruning variants in 10 minutes instead of 2 days.

## Installation

```bash
pip install -e .
```

## Quickstart

```python
from apl_pruning import MiniAPLParser
import numpy as np

parser = MiniAPLParser()
parser.set_variables(
    W=np.random.randn(768, 3072),
    act=np.random.randn(128, 768),
    grad=np.random.randn(768, 3072),
)

# Wanda in 1 line
scores = parser.evaluate("|W| x mean(|act|)")

# GPS^3: triple ratio
score = parser.evaluate("""
    rW <- max(|W|, dim=-1) / mean(|W|, dim=-1)
    rX <- max(mean(|act|, dim=0), dim=0) / mean(mean(|act|, dim=0))
    rG <- max(|grad|, dim=-1) / mean(|grad|, dim=-1)
    rW x rX x rG
""")
```

## Export to PyTorch

```python
from apl_pruning import to_pytorch, to_pytorch_function

print(to_pytorch("|W| x mean(|act|)"))
# (torch.abs(W) * torch.mean(torch.abs(act)))

print(to_pytorch_function("|W| x mean(|act|)", "wanda"))
# import torch
# def wanda(W, act):
#     return torch.abs(W) * torch.mean(torch.abs(act))
```

## 55 Built-in Methods

Core methods:

| Method | Formula | Needs Grad |
|--------|---------|------------|
| `magnitude` | `abs(W)` | No |
| `gradient` | `abs(W) * abs(grad)` | Yes |
| `wanda` | `abs(W) * mean(abs(act))` | No |
| `gps_local` | direction * selectivity * distortion | No |
| `gps_cube` | ratio(W) * ratio(X) * ratio(grad) | Yes |
| `gcs_score` | Wanda * (1 + irreplaceability) | No |
| `sparsegps_score` | Energy * Unicity | No |
| `softmax_grad_score` | softmax(abs(W)) * abs(grad) | Yes |
| `weighted_softmax_score` | softmax(abs(W)) * mean(abs(act)) | No |
| `chain_fc1` | abs(W1) * abs(grad1) * (1 + importance) | Yes |
| `chain_fc2` | abs(W2) * abs(grad2) | Yes |
| `latency_score` | 1 - abs(argmax(abs(X)) - T/2) / (T/2) | No |
| `symmetry_score` | 1 - abs(mean(A)-mean(B)) / (abs(mean(A))+abs(mean(B))) | No |
| `fractal_score` | min(1, max(0, abs(slope)/2)) | No |

Extended methods: `direction`, `selectivity`, `distortion`, `contrast_wanda`, `contrast_gradient`, `cos_w_x`, `cos_w_grad`, `gps_w_x`, `gps_w_grad`, `gps_cube`, `union_wanda_grad`, `union_all3`, `energy`, `otsu_mask`, and more.

See `apl_pruning/scorers.py` for the full list of 55 methods.

## Supported Primitives

### Unary Functions
`abs` `mean` `var` `std` `norm` `sum` `max` `min` `sqrt` `log` `exp` `softmax` `threshold` `topk` `rank` `sort`

All support `dim=` axis parameter.

### Binary Operators
`+` `-` `x` `/` `^`

### Indexing
`W[i]` `W[:, j]` `W[i, j]` `W[0:512]`

### Assignment
`scores <- expr`

## Safety

- Empty tensors raise `ValueError` (no silent NaN)
- Division by zero raises `ZeroDivisionError`
- `sqrt` of negative raises `ValueError`
- `log` of non-positive raises `ValueError`
- Dimension mismatch raises `ValueError` with shapes

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full compiler pipeline and module responsibilities.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new formula (3 levels: simple, multi-line, Python).

## Used by

- [CastNet v2](https://github.com/graybaud/castnet) — Sparse LLM inference (23 strategies)

## Run Tests

```bash
pytest tests/ -v
```

## License

MIT
