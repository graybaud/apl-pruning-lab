"""
APL Pruning Lab — Application Layer

This package orchestrates the domain (pure language) and infrastructure
(PyTorch export) to provide a complete pruning formula evaluation pipeline.

Architecture:
    domain/          — Pure language (tokenizer, grammar, evaluator, formulas)
    infrastructure/  — Export adapters (PyTorch, future: JAX)
    apl_pruning/     — Application (parser, layer scorer)  <-- YOU ARE HERE
    orchestration/   — Entry points (examples, CLI)
"""

__version__ = "0.5.0"

# Application layer (public API)
from apl_pruning.parser import MiniAPLParser
from apl_pruning.layers import LayerScorer

# Domain layer (re-exported for convenience)
from domain.scorers import (
    METHODS,
    score_layer,
    get_formula,
    get_pytorch,
    list_methods,
    compare_methods,
)
from domain.cache import LayerCache, fused_abs_mean, fused_abs_max, fused_abs_sum, fused_norm_ratio

# Infrastructure layer
from infrastructure.exporter import to_pytorch, to_pytorch_function

__all__ = [
    # Application
    "MiniAPLParser",
    "LayerScorer",
    # Domain
    "METHODS",
    "score_layer",
    "get_formula",
    "get_pytorch",
    "list_methods",
    "compare_methods",
    "LayerCache",
    "fused_abs_mean",
    "fused_abs_max",
    "fused_abs_sum",
    "fused_norm_ratio",
    # Infrastructure
    "to_pytorch",
    "to_pytorch_function",
]
