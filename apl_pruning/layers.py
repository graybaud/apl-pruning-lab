"""Layer-wise scoring: iterate over model layers automatically."""

import numpy as np
from apl_pruning import MiniAPLParser
from apl_pruning.cache import LayerCache


class LayerScorer:
    """Score pruning importance across all layers of a model.
    
    Example:
        layers = {
            "layer_0": {"W": W0, "act": act0},
            "layer_1": {"W": W1, "act": act1},
            ...
        }
        scorer = LayerScorer(layers)
        scores = scorer.score_all("|W| x mean(|act|)")
        for name, score in scores.items():
            print(f"{name}: {score.shape}")
    """
    
    def __init__(self, layers: dict):
        """
        Args:
            layers: dict of {layer_name: {var_name: numpy_array}}
        """
        self.layers = layers
        self.parser = MiniAPLParser()
        self.cache = LayerCache()
    
    def score_all(self, formula: str, use_cache: bool = True):
        """Evaluate formula on all layers.
        
        Args:
            formula: APL expression (e.g. "|W| x mean(|act|)")
            use_cache: If True, cache intermediate results per layer.
            
        Returns:
            dict: {layer_name: score_array}
        """
        results = {}
        for name, variables in self.layers.items():
            if use_cache:
                results[name] = self._score_with_cache(name, formula, variables)
            else:
                self.parser.set_variables(**variables)
                results[name] = self.parser.evaluate(formula)
        return results
    
    def _score_with_cache(self, layer_name: str, formula: str, variables: dict):
        """Evaluate with caching of common sub-expressions."""
        # Pre-compute common sub-expressions
        for var_name, value in variables.items():
            # Cache |X|
            abs_key = f"abs_{var_name}"
            if not self.cache.has(layer_name, abs_key):
                self.cache.store(layer_name, abs_key, np.abs(value))
            # Cache mean(|X|)
            mean_abs_key = f"mean_abs_{var_name}"
            if not self.cache.has(layer_name, mean_abs_key):
                self.cache.store(layer_name, mean_abs_key, np.mean(np.abs(value)))
            # Cache norm(X)
            norm_key = f"norm_{var_name}"
            if not self.cache.has(layer_name, norm_key):
                self.cache.store(layer_name, norm_key, np.linalg.norm(value))
        
        # Set variables with cached values
        all_vars = dict(variables)
        all_vars[f"|W|"] = self.cache.get(layer_name, "abs_W")
        all_vars[f"|act|"] = self.cache.get(layer_name, "abs_act")
        all_vars[f"mean(|W|)"] = self.cache.get(layer_name, "mean_abs_W")
        all_vars[f"mean(|act|)"] = self.cache.get(layer_name, "mean_abs_act")
        all_vars[f"norm(W)"] = self.cache.get(layer_name, "norm_W")
        all_vars[f"norm(act)"] = self.cache.get(layer_name, "norm_act")
        
        self.parser.set_variables(**all_vars)
        return self.parser.evaluate(formula)
    
    def compare(self, formulas: dict, use_cache: bool = True):
        """Compare multiple formulas across all layers.
        
        Args:
            formulas: dict of {method_name: formula}
            use_cache: If True, use caching for speed.
            
        Returns:
            dict: {method_name: {layer_name: score_array}}
        """
        results = {}
        for method_name, formula in formulas.items():
            results[method_name] = self.score_all(formula, use_cache=use_cache)
        return results
    
    def summary(self, scores: dict, top_k: int = 5):
        """Print a summary of scores across layers.
        
        Args:
            scores: dict from score_all() or compare()
            top_k: Number of layers to show
        """
        if isinstance(next(iter(scores.values())), dict):
            # Nested: compare() output
            for method_name, layer_scores in scores.items():
                print(f"\n{'='*60}")
                print(f"  Method: {method_name}")
                print(f"{'='*60}")
                self._print_summary(layer_scores, top_k)
        else:
            self._print_summary(scores, top_k)
    
    def _print_summary(self, scores: dict, top_k: int):
        for name, score in list(scores.items())[:top_k]:
            if hasattr(score, 'shape'):
                print(f"  {name:<20} shape={str(score.shape):<20} "
                      f"min={score.min():.4f} max={score.max():.4f} mean={score.mean():.4f}")
            else:
                print(f"  {name:<20} {score}")
        if len(scores) > top_k:
            print(f"  ... and {len(scores) - top_k} more layers")
