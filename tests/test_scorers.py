"""Tests for APL pruning method registry."""

import numpy as np
import pytest
from domain.scorers import (
    METHODS, score_layer, get_formula, get_pytorch, list_methods, compare_methods
)


@pytest.fixture
def data():
    return {
        "W": np.array([[1.0, -2.0, 3.0], [-0.5, 4.0, -1.0]]),
        "act": np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]]),
        "grad": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "S_full": np.eye(2),
        "S_without": np.array([[0.9, 0.1], [0.1, 0.85]]),
    }


def test_methods_defined():
    assert len(METHODS) >= 10
    for name, info in METHODS.items():
        assert "formula" in info
        assert "needs_grad" in info
        assert "description" in info
        assert "variables" in info


def test_score_magnitude(data):
    result = score_layer("magnitude", W=data["W"])
    expected = np.abs(data["W"])
    assert np.allclose(result, expected)


def test_score_wanda(data):
    result = score_layer("wanda", W=data["W"], act=data["act"])
    expected = np.abs(data["W"]) * np.mean(np.abs(data["act"]))
    assert np.allclose(result, expected)


def test_score_gradient(data):
    result = score_layer("gradient", W=data["W"], grad=data["grad"])
    expected = np.abs(data["W"]) * np.abs(data["grad"])
    assert np.allclose(result, expected)


def test_score_direction(data):
    result = score_layer("direction", W=data["W"])
    expected = np.max(np.abs(data["W"])) / np.mean(np.abs(data["W"]))
    assert np.allclose(result, expected)


def test_score_direction_per_neuron(data):
    result = score_layer("direction_per_neuron", W=data["W"])
    W = data["W"]
    expected = np.max(np.abs(W), axis=-1) / np.mean(np.abs(W), axis=-1)
    assert np.allclose(result, expected)


def test_score_selectivity(data):
    result = score_layer("selectivity", act=data["act"])
    expected = np.var(data["act"]) / np.mean(data["act"])
    assert np.allclose(result, expected)


def test_get_formula():
    formula = get_formula("wanda")
    assert "|W| x mean(|act|)" in formula


def test_get_pytorch():
    code = get_pytorch("wanda")
    assert "torch.abs" in code
    assert "torch.mean" in code


def test_list_methods():
    methods = list_methods()
    assert len(methods) >= 10
    assert any(m["name"] == "wanda" for m in methods)


def test_compare_methods(data):
    results = compare_methods(
        ["magnitude", "wanda", "gradient", "direction"],
        **data
    )
    assert "magnitude" in results
    assert "wanda" in results
    assert "gradient" in results
    assert "direction" in results


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        score_layer("nonexistent", W=np.array([1.0]))


def test_gps_formula(data):
    result = score_layer("gps", **data)
    W = data["W"]
    act = data["act"]
    Sf = data["S_full"]
    Sw = data["S_without"]
    d = np.max(np.abs(W), axis=-1) / np.mean(np.abs(W), axis=-1)
    s = np.var(act) / np.mean(act)
    dist = np.linalg.norm(Sf - Sw) / np.linalg.norm(Sf)
    # d is (2,) but s and dist are scalars, need broadcasting
    expected = d * s * dist
    assert np.allclose(result, expected, rtol=1e-4)


def test_threshold_method(data):
    result = score_layer("threshold", W=data["W"])
    expected = (np.abs(data["W"]) > 0.5).astype(np.float32)
    assert np.allclose(result, expected)
