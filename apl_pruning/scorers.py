"""Standard pruning methods expressed as APL formulas.

Each method is a dict with:
    formula: APL expression
    needs_grad: True if gradients required
    description: Human-readable name

Usage:
    from apl_pruning.scorers import METHODS, score_layer
    
    scores = score_layer("wanda", W=weights, act=activations)
    # or get the formula:
    formula = METHODS["wanda"]["formula"]
"""

from apl_pruning import MiniAPLParser

# ======================================================================
# Method Registry
# ======================================================================

METHODS = {
    "magnitude": {
        "formula": "|W|",
        "needs_grad": False,
        "description": "Weight magnitude |W|",
        "variables": ["W"],
    },
    "gradient": {
        "formula": "|W| x |grad|",
        "needs_grad": True,
        "description": "Gradient-weighted |W| * |grad|",
        "variables": ["W", "grad"],
    },
    "wanda": {
        "formula": "|W| x mean(|act|)",
        "needs_grad": False,
        "description": "Wanda: |W| * mean(|act|)",
        "variables": ["W", "act"],
    },
    "wanda_per_neuron": {
        "formula": "|W| x mean(|act|, dim=-1)",
        "needs_grad": False,
        "description": "Wanda per neuron: |W| * mean(|act|, dim=-1)",
        "variables": ["W", "act"],
    },
    "chain_grad": {
        "formula": "|W1| x |grad1| x |W2| x |grad2|",
        "needs_grad": True,
        "description": "Chain gradient: product of |W|*|grad| for paired layers",
        "variables": ["W1", "grad1", "W2", "grad2"],
    },
    "chain_wanda": {
        "formula": "|W1| x mean(|act1|) x |W2| x mean(|act2|)",
        "needs_grad": False,
        "description": "Chain Wanda: product for paired layers",
        "variables": ["W1", "act1", "W2", "act2"],
    },
    "softmax_grad": {
        "formula": "softmax(|W|, dim=-1) x |grad|",
        "needs_grad": True,
        "description": "Softmax(|W|) * |grad|",
        "variables": ["W", "grad"],
    },
    "weighted_softmax": {
        "formula": "softmax(|W|, dim=-1) x mean(|act|)",
        "needs_grad": False,
        "description": "Softmax(|W|) * mean(|act|)",
        "variables": ["W", "act"],
    },
    "direction": {
        "formula": "(max(|W|)) / mean(|W|)",
        "needs_grad": False,
        "description": "Direction: max(|W|) / mean(|W|)",
        "variables": ["W"],
    },
    "direction_per_neuron": {
        "formula": "max(|W|, dim=-1) / mean(|W|, dim=-1)",
        "needs_grad": False,
        "description": "Direction per neuron",
        "variables": ["W"],
    },
    "selectivity": {
        "formula": "var(act) / mean(act)",
        "needs_grad": False,
        "description": "Selectivity: var(act) / mean(act)",
        "variables": ["act"],
    },
    "distortion": {
        "formula": "norm(S_full - S_without) / norm(S_full)",
        "needs_grad": False,
        "description": "Distortion: ||S_full - S_without|| / ||S_full||",
        "variables": ["S_full", "S_without"],
    },
    "gps": {
        "formula": """
            direction <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            selectivity <- var(act) / mean(act)
            distortion <- norm(S_full - S_without) / norm(S_full)
            direction x selectivity x distortion
        """,
        "needs_grad": False,
        "description": "GPS: direction * selectivity * distortion",
        "variables": ["W", "act", "S_full", "S_without"],
    },
    "gps_local": {
        "formula": """
            direction <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            selectivity <- var(act) / mean(act)
            distortion <- norm(act_out) / norm(act_in)
            direction x selectivity x distortion
        """,
        "needs_grad": False,
        "description": "GPS local: direction * selectivity * distortion (local)",
        "variables": ["W", "act", "act_in", "act_out"],
    },
    "norm_ratio": {
        "formula": "norm(W, dim=-1) / (norm(grad, dim=-1) + 1e-8)",
        "needs_grad": True,
        "description": "Per-neuron norm ratio: ||W_i|| / ||grad_i||",
        "variables": ["W", "grad"],
    },
    "threshold": {
        "formula": "threshold(|W|, 0.5)",
        "needs_grad": False,
        "description": "Binary mask: |W| > 0.5",
        "variables": ["W"],
    },
}


# ======================================================================
# Scorer API
# ======================================================================

def score_layer(method: str, **variables):
    """Score a single layer using a named method.
    
    Args:
        method: One of METHODS keys (e.g. "wanda", "gradient", "gps")
        **variables: Tensors required by the method (W, act, grad, etc.)
    
    Returns:
        numpy array of scores
    
    Example:
        scores = score_layer("wanda", W=weights, act=activations)
    """
    if method not in METHODS:
        available = list(METHODS.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    
    formula = METHODS[method]["formula"]
    parser = MiniAPLParser()
    parser.set_variables(**variables)
    return parser.evaluate(formula)


def get_formula(method: str) -> str:
    """Get the APL formula for a method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'")
    return METHODS[method]["formula"]


def get_pytorch(method: str) -> str:
    """Get PyTorch code for a method."""
    from apl_pruning.exporter import to_pytorch
    formula = get_formula(method)
    return to_pytorch(formula)


def list_methods():
    """List all available methods."""
    return [
        {"name": k, "needs_grad": v["needs_grad"], "description": v["description"]}
        for k, v in METHODS.items()
    ]


def compare_methods(methods, **variables):
    """Compare multiple methods on the same data.
    
    Args:
        methods: List of method names
        **variables: All tensors needed by any method
    
    Returns:
        dict: {method_name: score_array}
    """
    results = {}
    parser = MiniAPLParser()
    for method in methods:
        if method not in METHODS:
            continue
        formula = METHODS[method]["formula"]
        # Only pass variables needed by this method
        needed = METHODS[method]["variables"]
        method_vars = {k: v for k, v in variables.items() if k in needed}
        parser.set_variables(**method_vars)
        results[method] = parser.evaluate(formula)
    return results
