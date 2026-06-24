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

METHODS.update({
    # === GPS Atomics ===
    "neuron_direction": {
        "formula": "max(|W|, dim=-1) / (mean(|W|, dim=-1) + 1e-8)",
        "needs_grad": False,
        "description": "GPS Direction: max/mean ratio per neuron",
        "variables": ["W"],
    },
    "neuron_selectivity": {
        "formula": "var(act, dim=0) / (mean(act, dim=0) + 1e-8)",
        "needs_grad": False,
        "description": "Selectivity: variance/mean per neuron",
        "variables": ["act"],
    },
    "cosine_similarity_matrix": {
        "formula": "normed @ normed.T",
        "needs_grad": False,
        "description": "Cosine similarity matrix (X must be pre-normalized)",
        "variables": ["normed"],
    },
    "frobenius_distortion": {
        "formula": "norm(S1 - S2) / (norm(S1) + 1e-8)",
        "needs_grad": False,
        "description": "Frobenius distortion between two similarity matrices",
        "variables": ["S1", "S2"],
    },
    
    # === GPS Chain V1 — Static weight propagation ===
    "gps_chain_v1_fc2": {
        "formula": """
            importance <- |fc1_W|.T @ gps_fc1
            importance <- importance / (max(importance) + 1e-8)
            gps_fc2 * (1.0 + importance)
        """,
        "needs_grad": False,
        "description": "GPS Chain V1: fc2 importance from downstream fc1 weights",
        "variables": ["gps_fc2", "gps_fc1", "fc1_W"],
    },
    "gps_chain_v1_fc1": {
        "formula": """
            importance <- |fc2_W|.T @ gps_fc2
            importance <- importance / (max(importance) + 1e-8)
            gps_fc1 * (1.0 + importance)
        """,
        "needs_grad": False,
        "description": "GPS Chain V1: fc1 importance from same-layer fc2 weights",
        "variables": ["gps_fc1", "gps_fc2", "fc2_W"],
    },
    
    # === GPS Chain V2 — Activation-weighted ===
    "gps_chain_v2_fc2": {
        "formula": """
            contrib <- act_fc2 x |fc1_W|
            importance <- sum(contrib * gps_fc1, dim=0)
            importance <- importance / (max(importance) + 1e-8)
            gps_fc2 * (1.0 + importance)
        """,
        "needs_grad": False,
        "description": "GPS Chain V2: fc2 importance from activations x weights",
        "variables": ["gps_fc2", "gps_fc1", "act_fc2", "fc1_W"],
    },
    
    # === GCS: Geometric Complement Scoring ===
    "gcs": {
        "formula": """
            wanda <- |W| x norm(X, dim=0)
            H_cos <- |normed_X.T @ normed_X|
            replacability <- max(min(H_cos, w_ratio), dim=1)
            irremplacability <- 1.0 - replacability
            wanda * (1.0 + irremplacability)
        """,
        "needs_grad": False,
        "description": "GCS: Wanda × (1 + irremplaçabilité)",
        "variables": ["W", "X", "normed_X", "w_ratio"],
    },
    
    # === SparseGPS ===
    "energy": {
        "formula": "mean(|X * W|, dim=0)",
        "needs_grad": False,
        "description": "Energy: mean(|W[k,j] * X[:,j]|)",
        "variables": ["W", "X"],
    },
    "unicity": {
        "formula": "1.0 - max(|corr_matrix|, dim=1)",
        "needs_grad": False,
        "description": "Unicity: 1 - max correlation with other connections",
        "variables": ["corr_matrix"],
    },
    "sparsegps": {
        "formula": """
            energy <- mean(|X * W|, dim=0)
            unicity <- 1.0 - max(|corr_matrix|, dim=1)
            energy * unicity
        """,
        "needs_grad": False,
        "description": "SparseGPS: Energy × Unicity",
        "variables": ["W", "X", "corr_matrix"],
    },
    
    # === Latency ===
    "latency": {
        "formula": "1.0 - |argmax(|X|, dim=0) - T/2| / (T/2)",
        "needs_grad": False,
        "description": "Latency score: 1.0 = activates mid-sequence",
        "variables": ["X", "T"],
    },
    
    # === Symmetry ===
    "symmetry": {
        "formula": "1.0 - |mean(A, dim=0) - mean(B, dim=0)| / (|mean(A, dim=0)| + |mean(B, dim=0)| + 1e-8)",
        "needs_grad": False,
        "description": "Symmetry score: invariance between two activation sets",
        "variables": ["A", "B"],
    },
    
    # === Fractal Dimension ===
    "fractal": {
        "formula": "min(1.0, max(0.0, |slope| / 2.0))",
        "needs_grad": False,
        "description": "Fractal score from log(var) vs log(scale) slope",
        "variables": ["slope"],
    },
    
    # === Logical Rules (R1, R3, R4, R6) ===
    "rule_r1_specialized": {
        "formula": "(direction > 4.0) & (grad_norm > 0.001)",
        "needs_grad": True,
        "description": "R1: Specialized AND task-sensitive",
        "variables": ["direction", "grad_norm"],
    },
    "rule_r3_unique": {
        "formula": "correlation < 0.4",
        "needs_grad": False,
        "description": "R3: Low correlation with peers",
        "variables": ["correlation"],
    },
    "rule_r4_impact": {
        "formula": "distortion > 0.001",
        "needs_grad": False,
        "description": "R4: High geometric impact",
        "variables": ["distortion"],
    },
    "rule_r6_dynamic": {
        "formula": "latency > 0.3",
        "needs_grad": False,
        "description": "R6: Strong temporal dynamics",
        "variables": ["latency"],
    },
    
    # === Unified (GPS³) ===
    "gps_cube": {
        "formula": """
            ratio_W <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            ratio_X <- max(|act|, dim=0) / mean(|act|, dim=0)
            ratio_grad <- max(|grad|, dim=-1) / mean(|grad|, dim=-1)
            ratio_W * ratio_X * ratio_grad
        """,
        "needs_grad": True,
        "description": "GPS³: ratio(W) × ratio(X) × ratio(∇L)",
        "variables": ["W", "act", "grad"],
    },
    
    # === Mask generators ===
    "otsu_threshold": {
        "formula": "threshold(scores, otsu_threshold)",
        "needs_grad": False,
        "description": "Otsu-based binary mask",
        "variables": ["scores", "otsu_threshold"],
    },
    "percentile_mask": {
        "formula": "scores >= quantile(scores, 1.0 - keep_frac)",
        "needs_grad": False,
        "description": "Top-k% percentile mask",
        "variables": ["scores", "keep_frac"],
    },
})

# ======================================================================
# Preprocessing
# ======================================================================

PREPROCESSING = {
    "normalize_minmax": {
        "formula": "X / (max(X) + 1e-8)",
        "description": "Min-max normalization to [0, 1]",
        "variables": ["X"],
    },
    "normalize_zscore": {
        "formula": "(X - mean(X)) / (std(X) + 1e-8)",
        "description": "Z-score normalization",
        "variables": ["X"],
    },
    "sparsity_pct": {
        "formula": "(1.0 - sum(X != 0) / numel(X)) * 100.0",
        "description": "Sparsity percentage",
        "variables": ["X"],
    },
    "dead_neuron_ratio": {
        "formula": "sum(sum(|W|, dim=-1) == 0) / shape(W)[0]",
        "description": "Fraction of dead neurons",
        "variables": ["W"],
    },
    "gamma_exponent": {
        "formula": "-slope",
        "description": "Power-law exponent from log-log fit",
        "variables": ["slope"],
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

# ======================================================================
# Extended methods (from CastNet extended_methods.py)
# ======================================================================

METHODS.update({
    # === Contrastes ===
    "contrast_wanda": {
        "formula": "max(|W| x mean(|act|, dim=0), dim=-1) / mean(|W| x mean(|act|, dim=0), dim=-1)",
        "needs_grad": False,
        "description": "Contrast Wanda: max/mean of |W| x ||X||",
        "variables": ["W", "act"],
    },
    "contrast_gradient": {
        "formula": "max(|W| x |grad|, dim=-1) / mean(|W| x |grad|, dim=-1)",
        "needs_grad": True,
        "description": "Contrast Gradient: max/mean of |W| x |grad|",
        "variables": ["W", "grad"],
    },

    # === Alignements cosinus ===
    "cos_w_x": {
        "formula": "sum(|W| x mean(|act|, dim=0), dim=-1) / (norm(|W|, dim=-1) x norm(mean(|act|, dim=0)) + 1e-8)",
        "needs_grad": False,
        "description": "Cos(W, X): weight-activation alignment",
        "variables": ["W", "act"],
    },
    "cos_w_grad": {
        "formula": "sum(|W| x |grad|, dim=-1) / (norm(|W|, dim=-1) x norm(|grad|, dim=-1) + 1e-8)",
        "needs_grad": True,
        "description": "Cos(W, grad): weight-gradient alignment",
        "variables": ["W", "grad"],
    },

    # === GPS doubles ===
    "gps_w_x": {
        "formula": "(max(|W|, dim=-1) / mean(|W|, dim=-1)) x (max(mean(|act|, dim=0), dim=0) / mean(mean(|act|, dim=0)))",
        "needs_grad": False,
        "description": "GPS WxX: ratio(W) x ratio(X)",
        "variables": ["W", "act"],
    },
    "gps_w_grad": {
        "formula": "(max(|W|, dim=-1) / mean(|W|, dim=-1)) x (max(|grad|, dim=-1) / mean(|grad|, dim=-1))",
        "needs_grad": True,
        "description": "GPS WxG: ratio(W) x ratio(grad)",
        "variables": ["W", "grad"],
    },

    # === GPS cube (triple ratio) ===
    "gps_cube": {
        "formula": """
            rW <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            rX <- max(mean(|act|, dim=0), dim=0) / mean(mean(|act|, dim=0))
            rG <- max(|grad|, dim=-1) / mean(|grad|, dim=-1)
            rW x rX x rG
        """,
        "needs_grad": True,
        "description": "GPS^3: ratio(W) x ratio(X) x ratio(grad)",
        "variables": ["W", "act", "grad"],
    },

    # === Fusions ===
    "union_wanda_grad": {
        "formula": "max(|W| x mean(|act|, dim=0), |W| x |grad|)",
        "needs_grad": True,
        "description": "Union Wanda+Gradient: max(wanda, gradient)",
        "variables": ["W", "act", "grad"],
    },
    "union_all3": {
        "formula": """
            wanda <- |W| x mean(|act|, dim=0)
            grad_score <- |W| x |grad|
            gps3 <- (max(|W|, dim=-1) / mean(|W|, dim=-1)) x (max(mean(|act|, dim=0), dim=0) / mean(mean(|act|, dim=0))) x (max(|grad|, dim=-1) / mean(|grad|, dim=-1))
            max(wanda, grad_score, gps3)
        """,
        "needs_grad": True,
        "description": "Union Wanda+Gradient+GPS^3",
        "variables": ["W", "act", "grad"],
    },

    # === SparseGPS (energy x unicity) ===
    "energy": {
        "formula": "mean(|X * W|, dim=0)",
        "needs_grad": False,
        "description": "Energy: mean(|W[k,j] * X[:,j]|)",
        "variables": ["W", "X"],
    },
    "sparsegps_score": {
        "formula": "mean(|X * W|, dim=0) x (1.0 - max(|corr_matrix|, dim=1))",
        "needs_grad": False,
        "description": "SparseGPS: Energy x Unicity",
        "variables": ["W", "X", "corr_matrix"],
    },

    # === GCS: Geometric Complement Scoring ===
    "gcs_score": {
        "formula": "|W| x norm(X, dim=0) x (1.0 + (1.0 - max(H_cos, dim=1)))",
        "needs_grad": False,
        "description": "GCS: Wanda x (1 + irreplaceability)",
        "variables": ["W", "X", "H_cos"],
    },

    # === Chain scoring ===
    "chain_fc1": {
        "formula": "|W1| x |grad1| x (1.0 + importance)",
        "needs_grad": True,
        "description": "Chain fc1: gradient with downstream importance bonus",
        "variables": ["W1", "grad1", "importance"],
    },
    "chain_fc2": {
        "formula": "|W2| x |grad2|",
        "needs_grad": True,
        "description": "Chain fc2: standard gradient",
        "variables": ["W2", "grad2"],
    },

    # === Latency (Q10) ===
    "latency_score": {
        "formula": "1.0 - |argmax(|X|, dim=0) - T/2| / (T/2)",
        "needs_grad": False,
        "description": "Latency: 1.0 = mid-sequence activation",
        "variables": ["X", "T"],
    },

    # === Symmetry (Q11) ===
    "symmetry_score": {
        "formula": "1.0 - |mean(A, dim=0) - mean(B, dim=0)| / (|mean(A, dim=0)| + |mean(B, dim=0)| + 1e-8)",
        "needs_grad": False,
        "description": "Symmetry: invariance between two activation sets",
        "variables": ["A", "B"],
    },

    # === Fractal (Q12) ===
    "fractal_score": {
        "formula": "min(1.0, max(0.0, |slope| / 2.0))",
        "needs_grad": False,
        "description": "Fractal dimension score from log(var) vs log(scale) slope",
        "variables": ["slope"],
    },

    # === Otsu threshold ===
    "otsu_mask": {
        "formula": "threshold(scores, threshold_value)",
        "needs_grad": False,
        "description": "Binary mask using Otsu threshold",
        "variables": ["scores", "threshold_value"],
    },

    # === Softmax variants ===
    "softmax_grad_score": {
        "formula": "softmax(|W|, dim=-1) x |grad|",
        "needs_grad": True,
        "description": "Softmax(|W|) x |grad|",
        "variables": ["W", "grad"],
    },
    "weighted_softmax_score": {
        "formula": "softmax(|W|, dim=-1) x mean(|act|, dim=0)",
        "needs_grad": False,
        "description": "Softmax(|W|) x mean(|act|)",
        "variables": ["W", "act"],
    },
})
