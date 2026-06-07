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


## Supported Primitives

### Unary Functions
| APL | Function | Axis support |
|-----|----------|:------------:|
| \|X\| | Absolute value | - |
| mean(X) | Mean | dim=-1, dim=0 |
| var(X) | Variance | dim=-1, dim=0 |
| std(X) | Standard deviation | dim=-1, dim=0 |
| norm(X) | L2 norm | dim=-1, dim=0 |
| sum(X) | Sum | dim=-1, dim=0 |
| max(X) | Maximum | dim=-1, dim=0 |
| min(X) | Minimum | dim=-1, dim=0 |
| sqrt(X) | Square root | - |
| log(X) | Natural log | - |
| exp(X) | Exponential | - |
| softmax(X) | Softmax | dim=0, dim=-1 |
| threshold(X, t) | Binary mask X > t | - |
| topk(X, k) | Top-k values | - |

### Binary Operators
| APL | Operation |
|-----|-----------|
| X + Y | Addition |
| X - Y | Subtraction |
| X x Y | Multiplication (with broadcasting) |
| X / Y | Division (raises on div-by-zero) |
| X ^ Y | Power |

### Indexing
| APL | Description |
|-----|-------------|
| W[i] | Row i |
| W[:, j] | Column j |
| W[i, j] | Element (i,j) |
| W[0:512] | Slice rows |

### Assignment
| APL | Description |
|-----|-------------|
| scores <- expr | Assign expression to variable |

## Safety

- Empty tensors raise `ValueError` (no silent NaN)
- Division by zero raises `ZeroDivisionError`
- sqrt of negative raises `ValueError`
- log of non-positive raises `ValueError`

## Use with LLMs

Give the LLM the docs/PRIMITIVES.md file and it writes your pruning formulas.

## Run Tests

pytest tests/ -v

## License

MIT






A. Le parseur niveau 2 : support des axes (dim=-1), slicing (W[i]), broadcasting, plus de primitives (softmax, std, topk, threshold)

B. Un benchmark automatique : tu files une liste de formules APL et un modèle, ça te sort un tableau comparatif (Wanda vs Grad×W vs ta méthode 3 composantes vs tout ce que tu veux)

C. L'intégration LLM : un prompt système + exemples pour que Claude/GPT écrive automatiquement les formules APL à partir d'un papier de recherche

D. La génération de code : le parseur peut exporter la version PyTorch équivalente pour vérification croisée

E. Un exemple complet sur un vrai modèle : Llama-7B, toutes les couches, avec des vraies métriques de pruning