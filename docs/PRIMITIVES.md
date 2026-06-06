# APL Primitives for Pruning DSL

Give this to your LLM. It will write pruning formulas.

## Unary Functions

| APL | Python/JAX | Description |
|-----|-----------|-------------|
| \|X\| | jnp.abs(X) | Absolute value |
| mean(X) | jnp.mean(X) | Mean of all elements |
| var(X) | jnp.var(X) | Variance |
| norm(X) | jnp.linalg.norm(X) | L2 norm |
| sum(X) | jnp.sum(X) | Sum of all elements |
| max(X) | jnp.max(X) | Maximum |
| min(X) | jnp.min(X) | Minimum |
| sqrt(X) | jnp.sqrt(X) | Square root |
| log(X) | jnp.log(X) | Natural log |
| exp(X) | jnp.exp(X) | Exponential |

## Binary Operators

| APL | Python/JAX | Description |
|-----|-----------|-------------|
| X + Y | X + Y | Addition |
| X - Y | X - Y | Subtraction |
| X x Y | X * Y | Multiplication |
| X / Y | X / Y | Division |
| X ^ Y | X ** Y | Power |

## Assignment

| APL | Description |
|-----|-------------|
| score <- expr | Assign expression to variable |

## Example Formulas

// Wanda
|W| x mean(|act|)

// Grad x Weight
|grad| x |W|

// Direction component
(max(|W|)) / mean(|W|)

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
