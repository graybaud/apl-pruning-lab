"""Tests for PyTorch exporter."""

import pytest
import numpy as np
from apl_pruning import MiniAPLParser
from infrastructure.exporter import to_pytorch, to_pytorch_function


@pytest.fixture
def parser():
    p = MiniAPLParser()
    p.set_variables(
        W=np.array([[1.0, -2.0, 3.0], [-0.5, 4.0, -1.0]]),
        act=np.array([[0.5, 0.3, 0.8], [0.1, 0.9, 0.2]]),
    )
    return p


def test_export_wanda():
    result = to_pytorch("|W| x mean(|act|)")
    expected = "(torch.abs(W) * torch.mean(torch.abs(act)))"
    assert result == expected


def test_export_axis():
    result = to_pytorch("mean(|W|, dim=-1)")
    expected = "torch.mean(torch.abs(W), dim=-1)"
    assert result == expected


def test_export_softmax():
    result = to_pytorch("softmax(W, dim=0)")
    expected = "torch.nn.functional.softmax(W, dim=0)"
    assert result == expected


def test_export_threshold():
    result = to_pytorch("threshold(|W|, 0.5)")
    expected = "(torch.abs(W) > 0.5).float()"
    assert result == expected


def test_export_division():
    result = to_pytorch("sum(|W|) / max(|W|)")
    expected = "(torch.sum(torch.abs(W)) / torch.max(torch.abs(W)))"
    assert result == expected


def test_export_indexing():
    result = to_pytorch("W[0]")
    expected = "W[0]"
    assert result == expected


def test_export_slice():
    result = to_pytorch("W[:, 1]")
    expected = "W[:, 1]"
    assert result == expected


def test_export_function():
    result = to_pytorch_function("|W| x mean(|act|)", "wanda")
    assert "import torch" in result
    assert "def wanda(W, act):" in result
    assert "torch.abs(W) * torch.mean(torch.abs(act))" in result


def test_parser_to_pytorch(parser):
    result = parser.to_pytorch("|W| x mean(|act|)")
    expected = "(torch.abs(W) * torch.mean(torch.abs(act)))"
    assert result == expected


def test_parser_to_pytorch_function(parser):
    result = parser.to_pytorch_function("|W| x mean(|act|)", "wanda")
    assert "def wanda" in result
