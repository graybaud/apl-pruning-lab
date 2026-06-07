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
| threshold(X, t) | (X > t).astype(float) | Binary mask |

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

## Additional Primitives (v0.2.1)

| APL | Python/NumPy | Description |
|-----|-------------|-------------|
| rank(X) | np.linalg.matrix_rank(X) | Matrix rank |
| sort(X) | np.sort(X) | Sort elements |

## Error Messages

The parser provides clear error messages:
- `Variable 'X' not defined. Available: [...]`
- `Cannot compute mean of empty tensor`
- `Division by near-zero. min(|denominator|) = ...`
- `Cannot compute sqrt of negative values. Use |X| first.`
- `Cannot compute log of non-positive values. Use |X| or X + epsilon.`
- `Dimension mismatch: (a,b) + (c,d) cannot be broadcast.`
