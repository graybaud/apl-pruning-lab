"""Tests for LayerScorer and LayerCache."""

import numpy as np
import pytest
from apl_pruning import LayerScorer, LayerCache


@pytest.fixture
def layers():
    return {
        "layer_0": {
            "W": np.array([[1.0, -2.0, 3.0], [-0.5, 4.0, -1.0]]),
            "act": np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]]),
        },
        "layer_1": {
            "W": np.array([[2.0, -1.0, 0.5], [3.0, -2.0, 1.0]]),
            "act": np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]]),
        },
    }


def test_layer_scorer_single(layers):
    scorer = LayerScorer(layers)
    scores = scorer.score_all("|W| x mean(|act|)")
    
    assert len(scores) == 2
    assert "layer_0" in scores
    assert "layer_1" in scores
    assert scores["layer_0"].shape == (2, 3)
    assert scores["layer_1"].shape == (2, 3)


def test_layer_scorer_compare(layers):
    scorer = LayerScorer(layers)
    methods = {"mag": "|W|", "wanda": "|W| x mean(|act|)"}
    results = scorer.compare(methods)
    
    assert "mag" in results
    assert "wanda" in results
    assert results["mag"]["layer_0"].shape == (2, 3)


def test_layer_scorer_no_cache(layers):
    scorer = LayerScorer(layers)
    scores_cached = scorer.score_all("|W| x mean(|act|)", use_cache=True)
    scores_nocache = scorer.score_all("|W| x mean(|act|)", use_cache=False)
    
    for name in scores_cached:
        assert np.allclose(scores_cached[name], scores_nocache[name])


def test_layer_cache():
    cache = LayerCache()
    
    cache.store("L0", "abs_W", np.array([1.0, 2.0]))
    assert cache.has("L0", "abs_W")
    assert not cache.has("L0", "nonexistent")
    
    val = cache.get("L0", "abs_W")
    assert np.allclose(val, np.array([1.0, 2.0]))
    assert cache.stats["hits"] == 1


def test_layer_cache_miss():
    cache = LayerCache()
    val = cache.get("L0", "missing")
    assert val is None
    assert cache.stats["misses"] == 1


def test_layer_scorer_per_neuron(layers):
    scorer = LayerScorer(layers)
    scores = scorer.score_all("mean(|W|, dim=-1)")
    
    assert scores["layer_0"].shape == (2,)
    assert scores["layer_1"].shape == (2,)
