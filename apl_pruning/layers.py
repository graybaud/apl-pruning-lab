"""Layer-wise scoring: iterate over model layers automatically."""

import numpy as np
from apl_pruning.parser import MiniAPLParser
from domain.cache import LayerCache


class LayerScorer:
    """Score pruning importance across all layers of a model.

    Example:
        layers = {
            "layer_0": {"W": W0, "act": act0},
            "layer_1": {"W": W1, "act": act1},
        }
        scorer = LayerScorer(layers)
        scores = scorer.score_all("|W| x mean(|act|)")
    """

    def __init__(self, layers: dict):
        self.layers = layers
        self.parser = MiniAPLParser()
        self.cache = LayerCache()

    def score_all(self, formula: str, use_cache: bool = True):
        results = {}
        for name, variables in self.layers.items():
            if use_cache:
                results[name] = self._score_with_cache(name, formula, variables)
            else:
                self.parser.set_variables(**variables)
                results[name] = self.parser.evaluate(formula)
        return results

    def _score_with_cache(self, layer_name: str, formula: str, variables: dict):
        for var_name, value in variables.items():
            abs_key = f"abs_{var_name}"
            if not self.cache.has(layer_name, abs_key):
                self.cache.store(layer_name, abs_key, np.abs(value))
            mean_abs_key = f"mean_abs_{var_name}"
            if not self.cache.has(layer_name, mean_abs_key):
                self.cache.store(layer_name, mean_abs_key, np.mean(np.abs(value)))
            norm_key = f"norm_{var_name}"
            if not self.cache.has(layer_name, norm_key):
                self.cache.store(layer_name, norm_key, np.linalg.norm(value))

        all_vars = dict(variables)
        for k in ["abs_W", "abs_act", "mean_abs_W", "mean_abs_act", "norm_W", "norm_act"]:
            cached = self.cache.get(layer_name, k)
            if cached is not None:
                all_vars[f"|{k.split('_',1)[1]}|"] = cached

        self.parser.set_variables(**all_vars)
        return self.parser.evaluate(formula)

    def compare(self, formulas: dict, use_cache: bool = True):
        results = {}
        for method_name, formula in formulas.items():
            results[method_name] = self.score_all(formula, use_cache=use_cache)
        return results

    def summary(self, scores: dict, top_k: int = 5):
        if isinstance(next(iter(scores.values())), dict):
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
