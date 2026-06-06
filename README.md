# APL Pruning Lab

Minimal APL-like DSL for describing LLM pruning methods. Compiles to JAX.
Test 10 pruning variants in 10 minutes instead of 2 days.
LLM-friendly: give it the primitives, it writes the formulas.

## Installation

pip install apl-pruning

## Quickstart

from apl_pruning import MiniAPLParser
import jax.numpy as jnp

parser = MiniAPLParser()
parser.set_variables(
    W=jnp.array([[1.0, -2.0], [3.0, -0.5]]),
    act=jnp.array([[0.5, 0.3], [0.1, 0.9]])
)

# Wanda in 1 line
scores = parser.evaluate("|W| x mean(|act|)")

## Supported Primitives

| APL | Function |
|-----|----------|
| \|X\| | Absolute value |
| mean(X) | Mean |
| var(X) | Variance |
| norm(X) | L2 norm |
| sum(X) | Sum |
| max(X) | Maximum |
| min(X) | Minimum |
| sqrt(X) | Square root |
| X + Y | Addition |
| X - Y | Subtraction |
| X x Y | Multiplication |
| X / Y | Division |
| X ^ Y | Power |

## Use with LLMs

Give the LLM the primitives list and it writes your pruning formulas.
