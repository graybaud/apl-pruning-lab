"""Tests for MiniAPLParser - Level 2 (numpy backend)."""

import numpy as np
import pytest
from apl_pruning import MiniAPLParser


def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


@pytest.fixture
def parser():
    p = MiniAPLParser()
    W = np.array([[1.0, -2.0, 3.0], [-0.5, 4.0, -1.0]])
    act = np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]])
    S_full = np.eye(2)
    S_without = np.array([[0.9, 0.1], [0.1, 0.85]])
    p.set_variables(W=W, act=act, S_full=S_full, S_without=S_without)
    return p


def test_wanda(parser):
    result = parser.evaluate("|W| x mean(|act|)")
    expected = np.abs(parser.variables["W"]) * np.mean(
        np.abs(parser.variables["act"])
    )
    assert np.allclose(result, expected)


def test_direction(parser):
    result = parser.evaluate("(max(|W|)) / mean(|W|)")
    W = parser.variables["W"]
    expected = np.max(np.abs(W)) / np.mean(np.abs(W))
    assert np.allclose(result, expected)


def test_distortion(parser):
    result = parser.evaluate("norm(S_full - S_without) / norm(S_full)")
    Sf = parser.variables["S_full"]
    Sw = parser.variables["S_without"]
    expected = np.linalg.norm(Sf - Sw) / np.linalg.norm(Sf)
    assert np.allclose(result, expected)


def test_three_components(parser):
    result = parser.evaluate(
        "direction <- (max(|W|)) / mean(|W|)\n"
        "selectivity <- var(act) / mean(act)\n"
        "distortion <- norm(S_full - S_without) / norm(S_full)\n"
        "direction x selectivity x distortion"
    )
    W = parser.variables["W"]
    act = parser.variables["act"]
    Sf = parser.variables["S_full"]
    Sw = parser.variables["S_without"]
    d = np.max(np.abs(W)) / np.mean(np.abs(W))
    s = np.var(act) / np.mean(act)
    dist = np.linalg.norm(Sf - Sw) / np.linalg.norm(Sf)
    expected = d * s * dist
    assert np.allclose(result, expected)


def test_normalized_scores(parser):
    result = parser.evaluate(
        "scores <- |W| x mean(|act|)\n"
        "scores / sum(scores)"
    )
    assert np.allclose(np.sum(result), 1.0)


def test_mean_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("mean(|W|, dim=-1)")
    expected = np.mean(np.abs(W), axis=-1)
    assert np.allclose(result, expected)
    assert result.shape == (2,)


def test_sum_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("sum(|W|, dim=0)")
    expected = np.sum(np.abs(W), axis=0)
    assert np.allclose(result, expected)
    assert result.shape == (3,)


def test_norm_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("norm(W, dim=-1)")
    expected = np.linalg.norm(W, axis=-1)
    assert np.allclose(result, expected)
    assert result.shape == (2,)


def test_index_row(parser):
    result = parser.evaluate("W[0]")
    expected = parser.variables["W"][0]
    assert np.allclose(result, expected)


def test_index_column(parser):
    result = parser.evaluate("W[:, 1]")
    expected = parser.variables["W"][:, 1]
    assert np.allclose(result, expected)


def test_index_element(parser):
    result = parser.evaluate("W[0, 1]")
    expected = parser.variables["W"][0, 1]
    assert np.allclose(result, expected)


def test_std(parser):
    result = parser.evaluate("std(W)")
    expected = np.std(parser.variables["W"])
    assert np.allclose(result, expected)


def test_softmax(parser):
    result = parser.evaluate("softmax(W)")
    expected = _softmax(parser.variables["W"], axis=-1)
    assert np.allclose(result, expected)


def test_threshold(parser):
    result = parser.evaluate("threshold(|W|, 2.0)")
    expected = (np.abs(parser.variables["W"]) > 2.0).astype(np.float32)
    assert np.allclose(result, expected)


def test_wanda_per_neuron(parser):
    result = parser.evaluate("|W| x mean(|act|)")
    expected = np.abs(parser.variables["W"]) * np.mean(
        np.abs(parser.variables["act"])
    )
    assert np.allclose(result, expected)


def test_variable_not_found(parser):
    with pytest.raises(NameError):
        parser.evaluate("mean(Z)")


# New edge case tests
def test_broadcasting_multiply(parser):
    vec = np.array([2.0, 3.0])
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    parser.set_variables(vec=vec, mat=mat)
    result = parser.evaluate("vec x mat")


def test_nested_abs(parser):
    result = parser.evaluate("| |W| |")
    expected = np.abs(np.abs(parser.variables["W"]))
    assert np.allclose(result, expected)


def test_chain_operations(parser):
    result = parser.evaluate("sum(|W|) / (max(|W|) + 0.001)")
    W = parser.variables["W"]
    expected = np.sum(np.abs(W)) / (np.max(np.abs(W)) + 0.001)
    assert np.allclose(result, expected)


def test_zero_tensor(parser):
    Z = np.zeros((3, 4))
    parser.set_variable('Z', Z)
    result = parser.evaluate("mean(Z)")
    assert result == 0.0


def test_single_element(parser):
    s = np.array([42.0])
    parser.set_variable('s', s)
    result = parser.evaluate("s")
    assert result == 42.0


def test_softmax_axis_param(parser):
    result = parser.evaluate("softmax(W, dim=0)")
    expected = _softmax(parser.variables["W"], axis=0)
    assert np.allclose(result, expected)
