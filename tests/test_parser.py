"""Tests for MiniAPLParser - Level 2."""

import jax.numpy as jnp
import jax
import pytest
from apl_pruning import MiniAPLParser


@pytest.fixture
def parser():
    p = MiniAPLParser()
    W = jnp.array([[1.0, -2.0, 3.0], [-0.5, 4.0, -1.0]])
    act = jnp.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]])
    S_full = jnp.eye(2)
    S_without = jnp.array([[0.9, 0.1], [0.1, 0.85]])
    p.set_variables(W=W, act=act, S_full=S_full, S_without=S_without)
    return p


# Level 1 tests (should still pass)
def test_wanda(parser):
    result = parser.evaluate("|W| x mean(|act|)")
    expected = jnp.abs(parser.variables["W"]) * jnp.mean(
        jnp.abs(parser.variables["act"])
    )
    assert jnp.allclose(result, expected)


def test_direction(parser):
    result = parser.evaluate("(max(|W|)) / mean(|W|)")
    W = parser.variables["W"]
    expected = jnp.max(jnp.abs(W)) / jnp.mean(jnp.abs(W))
    assert jnp.allclose(result, expected)


def test_distortion(parser):
    result = parser.evaluate("norm(S_full - S_without) / norm(S_full)")
    Sf = parser.variables["S_full"]
    Sw = parser.variables["S_without"]
    expected = jnp.linalg.norm(Sf - Sw) / jnp.linalg.norm(Sf)
    assert jnp.allclose(result, expected)


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
    d = jnp.max(jnp.abs(W)) / jnp.mean(jnp.abs(W))
    s = jnp.var(act) / jnp.mean(act)
    dist = jnp.linalg.norm(Sf - Sw) / jnp.linalg.norm(Sf)
    expected = d * s * dist
    assert jnp.allclose(result, expected)


def test_normalized_scores(parser):
    result = parser.evaluate(
        "scores <- |W| x mean(|act|)\n"
        "scores / sum(scores)"
    )
    assert jnp.allclose(jnp.sum(result), 1.0)


# Level 2 tests - axis support
def test_mean_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("mean(|W|, dim=-1)")
    expected = jnp.mean(jnp.abs(W), axis=-1)
    assert jnp.allclose(result, expected)
    assert result.shape == (2,)


def test_sum_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("sum(|W|, dim=0)")
    expected = jnp.sum(jnp.abs(W), axis=0)
    assert jnp.allclose(result, expected)
    assert result.shape == (3,)


def test_norm_along_axis(parser):
    W = parser.variables["W"]
    result = parser.evaluate("norm(W, dim=-1)")
    expected = jnp.linalg.norm(W, axis=-1)
    assert jnp.allclose(result, expected)
    assert result.shape == (2,)


# Level 2 tests - indexing
def test_index_row(parser):
    result = parser.evaluate("W[0]")
    expected = parser.variables["W"][0]
    assert jnp.allclose(result, expected)


def test_index_column(parser):
    result = parser.evaluate("W[:, 1]")
    expected = parser.variables["W"][:, 1]
    assert jnp.allclose(result, expected)


def test_index_element(parser):
    result = parser.evaluate("W[0, 1]")
    expected = parser.variables["W"][0, 1]
    assert jnp.allclose(result, expected)


# Level 2 tests - new primitives
def test_std(parser):
    result = parser.evaluate("std(W)")
    expected = jnp.std(parser.variables["W"])
    assert jnp.allclose(result, expected)


def test_softmax(parser):
    result = parser.evaluate("softmax(W)")
    expected = jax.nn.softmax(parser.variables["W"], axis=-1)
    assert jnp.allclose(result, expected)


def test_threshold(parser):
    result = parser.evaluate("threshold(|W|, 2.0)")
    expected = (jnp.abs(parser.variables["W"]) > 2.0).astype(jnp.float32)
    assert jnp.allclose(result, expected)


def test_wanda_per_neuron(parser):
    """|W| x mean(|act|, dim=-1) -> score per neuron"""
    result = parser.evaluate("|W| x mean(|act|, dim=-1)")
    # mean(|act|, dim=-1) -> shape (2,), broadcasts with W (2,3)
    expected = jnp.abs(parser.variables["W"]) * jnp.mean(
        jnp.abs(parser.variables["act"]), axis=-1, keepdims=True
    ).reshape(-1, 1)
    assert jnp.allclose(result, expected)


# Level 2 tests - error handling
def test_variable_not_found(parser):
    with pytest.raises(NameError):
        parser.evaluate("mean(Z)")


def test_index_out_of_bounds(parser):
    with pytest.raises(IndexError):
        parser.evaluate("W[100]")
