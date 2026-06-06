"""Tests for MiniAPLParser."""

import jax.numpy as jnp
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


def test_variable_not_found(parser):
    with pytest.raises(NameError):
        parser.evaluate("mean(Z)")


def test_normalized_scores(parser):
    result = parser.evaluate(
        "scores <- |W| x mean(|act|)\n"
        "scores / sum(scores)"
    )
    assert jnp.allclose(jnp.sum(result), 1.0)
