# APL Pruning Lab

Minimal APL-like DSL for describing LLM pruning methods. Compiles to numpy.
Test 10 pruning variants in 10 minutes instead of 2 days.
LLM-friendly: give it the primitives, it writes the formulas.

## Installation


pip install -e .


## Quickstart


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

# Three-component score
score = parser.evaluate("""
    direction <- (max(|W|)) / mean(|W|)
    selectivity <- var(act) / mean(act)
    distortion <- norm(S_full - S_without) / norm(S_full)
    direction x selectivity x distortion
""")


## Export to PyTorch


from apl_pruning import to_pytorch, to_pytorch_function

print(to_pytorch("|W| x mean(|act|)"))
# (torch.abs(W) * torch.mean(torch.abs(act)))

print(to_pytorch_function("|W| x mean(|act|)", "wanda"))
# import torch
# def wanda(W, act):
#     return torch.abs(W) * torch.mean(torch.abs(act))


## Supported Primitives

### Unary Functions (all support `dim=` axis parameter)
`abs`, `mean`, `var`, `std`, `norm`, `sum`, `max`, `min`, `sqrt`, `log`, `exp`, `softmax`, `threshold`, `topk`, `rank`, `sort`

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


apl_pruning/
├── __init__.py          # Public API, version
├── parser.py            # Main class (orchestration)
├── tokenizer.py         # Lexer: code -> tokens
├── grammar.py           # Parser: tokens -> AST
├── ast_evaluator.py     # Evaluator: AST -> numpy
├── exporter.py          # AST -> PyTorch code
└── utils.py             # Safe math, broadcasting


## Run Tests


pytest tests/ -v


## Use with LLMs

Give the LLM `docs/PRIMITIVES.md` and it writes your pruning formulas.

## License

MIT
