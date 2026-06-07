# APL Primitives for Pruning DSL

Give this to your LLM. It will write pruning formulas.

## Unary Functions

| APL | Python/NumPy | Description |
|-----|-------------|-------------|
| \|X\| | np.abs(X) | Absolute value |
| mean(X) | np.mean(X) | Mean of all elements |
| mean(X, dim=-1) | np.mean(X, axis=-1) | Mean along last axis |
| var(X) | np.var(X) | Variance |
| std(X) | np.std(X) | Standard deviation |
| norm(X) | np.linalg.norm(X) | L2 norm |
| norm(X, dim=-1) | np.linalg.norm(X, axis=-1) | Per-vector norm |
| sum(X) | np.sum(X) | Sum of all elements |
| sum(X, dim=0) | np.sum(X, axis=0) | Sum along first axis |
| max(X) | np.max(X) | Maximum |
| min(X) | np.min(X) | Minimum |
| sqrt(X) | np.sqrt(X) | Square root |
| log(X) | np.log(X) | Natural log |
| exp(X) | np.exp(X) | Exponential |
| softmax(X) | softmax(X) | Softmax along last axis |
| softmax(X, dim=0) | softmax(X, axis=0) | Softmax along first axis |
| threshold(X, t) | (X > t).float() | Binary mask |
| topk(X, k) | topk(X.flatten(), k).values | Top-k values |
| rank(X) | matrix_rank(X) | Matrix rank |
| sort(X) | np.sort(X) | Sort elements |

## Binary Operators

| APL | Python/NumPy | Description |
|-----|-------------|-------------|
| X + Y | X + Y | Addition |
| X - Y | X - Y | Subtraction |
| X x Y | X * Y | Multiplication |
| X / Y | X / Y | Division |
| X ^ Y | X ** Y | Power |

## Assignment

| APL | Description |
|-----|-------------|
| score <- expr | Assign expression to variable |

## Indexing

| APL | NumPy | Description |
|-----|-------|-------------|
| W[i] | W[i] | Row i |
| W[:, j] | W[:, j] | Column j |
| W[i, j] | W[i, j] | Element (i,j) |
| W[0:512] | W[0:512] | Slice rows |

## PyTorch Export

Any formula can be exported to PyTorch:


from apl_pruning import to_pytorch, to_pytorch_function

# Single expression
print(to_pytorch("|W| x mean(|act|)"))
# -> (torch.abs(W) * torch.mean(torch.abs(act)))

# Complete function
print(to_pytorch_function("|W| x mean(|act|)", "wanda"))
# -> import torch
#    def wanda(W, act):
#        return torch.abs(W) * torch.mean(torch.abs(act))


## Error Messages

- `Variable 'X' not defined. Available: [...]`
- `Cannot compute mean of empty tensor`
- `Division by near-zero. min(|denominator|) = ...`
- `Cannot compute sqrt of negative values. Use |X| first.`
- `Cannot compute log of non-positive values. Use |X| or X + epsilon.`
- `Dimension mismatch: (a,b) + (c,d) cannot be broadcast.`

## Example Formulas


// Wanda
|W| x mean(|act|)

// Grad x Weight
|grad| x |W|

// Direction component (scalar)
(max(|W|)) / mean(|W|)

// Direction per neuron
max(|W|, dim=-1) / mean(|W|, dim=-1)

// Selectivity component
var(act) / mean(act)

// Distortion component
norm(S_full - S_without) / norm(S_full)

// Three-component score
direction <- (max(|W|)) / mean(|W|)
selectivity <- var(act) / mean(act)
distortion <- norm(S_full - S_without) / norm(S_full)
direction x selectivity x distortion

// Normalized scores
scores <- |W| x mean(|act|)
scores / sum(scores)

// Per-neuron Wanda
|W| x mean(|act|, dim=-1)

// Softmax of weight magnitudes
softmax(|W|)

// Threshold mask
threshold(|W|, 0.5)

// Top-10 weights
topk(|W|, 10)

// Matrix rank
rank(W)

