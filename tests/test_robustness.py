"""Robustness tests: shapes, broadcasting, edge cases, fuzzing, regression."""

import numpy as np
import pytest
from apl_pruning import MiniAPLParser


@pytest.fixture
def parser():
    p = MiniAPLParser()
    rng = np.random.RandomState(42)
    p.set_variables(
        W=rng.randn(768, 3072).astype(np.float32),
        act=rng.randn(128, 768).astype(np.float32),
        grad=rng.randn(768, 3072).astype(np.float32),
    )
    return p


class TestShapes:
    def test_abs_preserves_shape(self, parser):
        result = parser.evaluate("|W|")
        assert result.shape == parser.variables["W"].shape

    def test_mean_scalar(self, parser):
        result = parser.evaluate("mean(W)")
        assert np.ndim(result) == 0

    def test_mean_along_axis_shape(self, parser):
        result = parser.evaluate("mean(|W|, dim=-1)")
        assert result.shape == (768,)

    def test_sum_along_axis_shape(self, parser):
        result = parser.evaluate("sum(|W|, dim=0)")
        assert result.shape == (3072,)

    def test_norm_along_axis_shape(self, parser):
        result = parser.evaluate("norm(W, dim=-1)")
        assert result.shape == (768,)

    def test_std_preserves_scalar(self, parser):
        result = parser.evaluate("std(W)")
        assert np.ndim(result) == 0

    def test_softmax_preserves_shape(self, parser):
        result = parser.evaluate("softmax(W)")
        assert result.shape == parser.variables["W"].shape

    def test_threshold_preserves_shape(self, parser):
        result = parser.evaluate("threshold(|W|, 0.5)")
        assert result.shape == parser.variables["W"].shape

    def test_multiplication_same_shape(self, parser):
        result = parser.evaluate("|W| x |grad|")
        assert result.shape == (768, 3072)

    def test_addition_same_shape(self, parser):
        result = parser.evaluate("W + grad")
        assert result.shape == (768, 3072)

    def test_wanda_per_neuron_shape(self, parser):
        result = parser.evaluate("|W| x mean(|act|)")
        assert result.shape == (768, 3072)


class TestBroadcasting:
    def test_vector_times_matrix(self, parser):
        vec = np.array([2.0, 3.0])
        mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        parser.set_variables(vec=vec, mat=mat)
        result = parser.evaluate("vec x mat")
        expected = vec.reshape(-1, 1) * mat
        assert result.shape == (2, 3)
        assert np.allclose(result, expected)

    def test_scalar_times_matrix(self, parser):
        s = np.array(5.0)
        mat = np.ones((3, 4))
        parser.set_variables(s=s, mat=mat)
        result = parser.evaluate("s x mat")
        assert result.shape == (3, 4)

    def test_scalar_plus_matrix(self, parser):
        s = np.array(3.0)
        mat = np.ones((2, 5))
        parser.set_variables(s=s, mat=mat)
        result = parser.evaluate("s + mat")
        assert np.allclose(result, np.ones((2, 5)) + 3.0)

    def test_mean_along_axis_times_matrix(self, parser):
        result = parser.evaluate("|W| x mean(|W|, dim=-1)")
        assert result.shape == (768, 3072)

    def test_norm_ratio_broadcast(self, parser):
        result = parser.evaluate("norm(W, dim=-1) / norm(grad, dim=-1)")
        assert result.shape == (768,)


class TestEdgeCases:
    def test_empty_tensor_raises(self):
        p = MiniAPLParser()
        p.set_variable('E', np.array([]))
        with pytest.raises(ValueError, match="empty"):
            p.evaluate("mean(E)")

    def test_empty_tensor_sum_returns_zero(self):
        p = MiniAPLParser()
        p.set_variable('E', np.array([]))
        result = p.evaluate("sum(E)")
        assert result == 0.0

    def test_empty_tensor_max_raises(self):
        p = MiniAPLParser()
        p.set_variable('E', np.array([]))
        with pytest.raises(ValueError, match="empty"):
            p.evaluate("max(E)")

    def test_single_element(self):
        p = MiniAPLParser()
        p.set_variable('s', np.array([42.0]))
        result = p.evaluate("s")
        assert result == 42.0

    def test_single_element_abs(self):
        p = MiniAPLParser()
        p.set_variable('s', np.array([-7.0]))
        result = p.evaluate("|s|")
        assert result == 7.0

    def test_all_zeros(self, parser):
        Z = np.zeros((10, 10))
        parser.set_variable('Z', Z)
        result = parser.evaluate("mean(Z)")
        assert result == 0.0

    def test_all_zeros_std(self, parser):
        Z = np.zeros((5, 5))
        parser.set_variable('Z', Z)
        result = parser.evaluate("std(Z)")
        assert result == 0.0

    def test_all_zeros_norm(self, parser):
        Z = np.zeros((3, 3))
        parser.set_variable('Z', Z)
        result = parser.evaluate("norm(Z)")
        assert result == 0.0

    def test_division_by_near_zero_raises(self):
        p = MiniAPLParser()
        p.set_variables(a=np.array([1.0]), b=np.array([0.0]))
        with pytest.raises(ZeroDivisionError):
            p.evaluate("a / b")

    def test_division_ok_with_nonzero(self, parser):
        result = parser.evaluate("sum(|W|) / (max(|W|) + 1.0)")
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_softmax_all_zeros(self):
        p = MiniAPLParser()
        p.set_variable('Z', np.zeros((3, 4)))
        result = p.evaluate("softmax(Z)")
        expected = np.ones((3, 4)) / 4
        assert np.allclose(result, expected)

    def test_very_large_values(self):
        p = MiniAPLParser()
        p.set_variable('X', np.array([1e10, -1e10, 0.0]))
        result = p.evaluate("softmax(X)")
        assert np.allclose(np.sum(result), 1.0)
        assert not np.any(np.isnan(result))

    def test_negative_values_in_log_raises(self):
        p = MiniAPLParser()
        p.set_variable('X', np.array([-1.0, -2.0]))
        with pytest.raises(ValueError, match="log"):
            p.evaluate("log(X)")

    def test_log_of_abs_works(self):
        p = MiniAPLParser()
        p.set_variable('X', np.array([-1.0, -2.0]))
        result = p.evaluate("log(|X|)")
        expected = np.log(np.abs(np.array([-1.0, -2.0])))
        assert np.allclose(result, expected)

    def test_sqrt_of_negative_raises(self):
        p = MiniAPLParser()
        p.set_variable('X', np.array([-1.0]))
        with pytest.raises(ValueError, match="sqrt"):
            p.evaluate("sqrt(X)")

    def test_one_dimensional_tensor(self):
        p = MiniAPLParser()
        p.set_variable('v', np.array([1.0, 2.0, 3.0, 4.0]))
        result = p.evaluate("mean(v)")
        assert result == 2.5

    def test_two_dimensional_single_row(self):
        p = MiniAPLParser()
        p.set_variable('M', np.array([[1.0, 2.0, 3.0]]))
        result = p.evaluate("mean(M, dim=-1)")
        assert result.shape == (1,)
        assert result[0] == 2.0


class TestRegression:
    def test_wanda_regression(self, parser):
        result = parser.evaluate("|W| x mean(|act|)")
        W, act = parser.variables["W"], parser.variables["act"]
        expected = np.abs(W) * np.mean(np.abs(act))
        assert np.allclose(result, expected)

    def test_grad_weight_regression(self, parser):
        result = parser.evaluate("|grad| x |W|")
        W, grad = parser.variables["W"], parser.variables["grad"]
        expected = np.abs(grad) * np.abs(W)
        assert np.allclose(result, expected)

    def test_direction_regression(self, parser):
        result = parser.evaluate("(max(|W|)) / mean(|W|)")
        W = parser.variables["W"]
        expected = np.max(np.abs(W)) / np.mean(np.abs(W))
        assert np.allclose(result, expected)

    def test_selectivity_regression(self, parser):
        result = parser.evaluate("var(act) / mean(act)")
        act = parser.variables["act"]
        expected = np.var(act) / np.mean(act)
        assert np.allclose(result, expected)

    def test_per_neuron_mean_regression(self, parser):
        result = parser.evaluate("mean(|W|, dim=-1)")
        W = parser.variables["W"]
        expected = np.mean(np.abs(W), axis=-1)
        assert np.allclose(result, expected)
        assert result.shape == expected.shape

    def test_per_neuron_std_regression(self, parser):
        result = parser.evaluate("std(W, dim=-1)")
        W = parser.variables["W"]
        expected = np.std(W, axis=-1)
        assert np.allclose(result, expected)

    def test_combined_score_regression(self, parser):
        result = parser.evaluate("""
            direction <- (max(|W|)) / mean(|W|)
            selectivity <- var(act) / mean(act)
            norm_score <- norm(W) / (norm(grad) + 1.0)
            direction x selectivity x norm_score
        """)
        W = parser.variables["W"]
        act = parser.variables["act"]
        grad = parser.variables["grad"]
        d = np.max(np.abs(W)) / np.mean(np.abs(W))
        s = np.var(act) / np.mean(act)
        n = np.linalg.norm(W) / (np.linalg.norm(grad) + 1.0)
        expected = d * s * n
        assert np.allclose(result, expected)

    def test_softmax_regression(self, parser):
        result = parser.evaluate("softmax(W[0])")
        W0 = parser.variables["W"][0]
        e = np.exp(W0 - np.max(W0))
        expected = e / np.sum(e)
        assert np.allclose(result, expected)

    def test_chained_operations_regression(self, parser):
        W = parser.variables["W"]
        result = parser.evaluate("sum(|W|) / (max(|W|) + 1.0)")
        expected = np.sum(np.abs(W)) / (np.max(np.abs(W)) + 1.0)
        assert np.allclose(result, expected)

    def test_nested_abs_regression(self, parser):
        W = parser.variables["W"]
        result = parser.evaluate("| (|W| - 0.5) |")
        expected = np.abs(np.abs(W) - 0.5)
        assert np.allclose(result, expected)


class TestFuzzing:
    def test_random_unary_operations(self, parser):
        funcs = ['abs', 'mean', 'var', 'std', 'norm', 'sum', 'max', 'min']
        np_funcs = {
            'abs': np.abs, 'mean': np.mean, 'var': np.var, 'std': np.std,
            'norm': np.linalg.norm, 'sum': np.sum, 'max': np.max, 'min': np.min
        }
        for f in funcs:
            code = f"{f}(|W|)"
            result = parser.evaluate(code)
            W_abs = np.abs(parser.variables["W"])
            expected = np_funcs[f](W_abs)
            assert np.allclose(result, expected, rtol=1e-5), f"Mismatch for {f}"

    def test_random_binary_operations(self, parser):
        ops = {'+': np.add, '-': np.subtract, 'x': np.multiply, '^': np.power}
        W = parser.variables["W"]
        grad = parser.variables["grad"]
        for op, np_op in ops.items():
            code = f"|W| {op} |grad|"
            result = parser.evaluate(code)
            expected = np_op(np.abs(W), np.abs(grad))
            assert np.allclose(result, expected, rtol=1e-5), f"Mismatch for {op}"

    def test_safe_division(self, parser):
        W = parser.variables["W"]
        grad = parser.variables["grad"]
        result = parser.evaluate("|W| / (|grad| + 1.0)")
        expected = np.abs(W) / (np.abs(grad) + 1.0)
        assert np.allclose(result, expected, rtol=1e-5)

    def test_random_complex_expressions(self, parser):
        W = parser.variables["W"]
        test_cases = [
            ("max(|W|) - min(|W|)",
             lambda w: np.max(np.abs(w)) - np.min(np.abs(w))),
            ("sum(|W|) / (mean(|W|) + 1.0)",
             lambda w: np.sum(np.abs(w)) / (np.mean(np.abs(w)) + 1.0)),
            ("norm(W, dim=-1) / (std(W, dim=-1) + 1.0)",
             lambda w: np.linalg.norm(w, axis=-1) / (np.std(w, axis=-1) + 1.0)),
        ]
        for code, expected_fn in test_cases:
            result = parser.evaluate(code)
            expected = expected_fn(W)
            assert np.allclose(result, expected, rtol=1e-5), f"Mismatch for: {code}"

    def test_expression_with_all_primitives(self, parser):
        W = parser.variables["W"]
        grad = parser.variables["grad"]
        act = parser.variables["act"]
        
        result = parser.evaluate("""
            a <- mean(|W|)
            b <- std(W)
            c <- max(|grad|)
            d <- norm(W, dim=-1)
            e <- sum(|act|)
            (a x c + b) / (mean(d) + e + 1.0)
        """)
        
        a = np.mean(np.abs(W))
        b = np.std(W)
        c = np.max(np.abs(grad))
        d = np.linalg.norm(W, axis=-1)
        e = np.sum(np.abs(act))
        expected = (a * c + b) / (np.mean(d) + e + 1.0)
        assert np.allclose(result, expected, rtol=1e-5)

    def test_indexing_then_operation(self, parser):
        result = parser.evaluate("mean(|W[0]|)")
        expected = np.mean(np.abs(parser.variables["W"][0]))
        assert np.allclose(result, expected)

    def test_slice_then_operation(self, parser):
        result = parser.evaluate("sum(W[:, 0:100])")
        expected = np.sum(parser.variables["W"][:, 0:100])
        assert np.allclose(result, expected)


class TestNewPrimitives:
    def test_rank_matrix(self):
        p = MiniAPLParser()
        p.set_variable('M', np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = p.evaluate("rank(M)")
        assert result == 2

    def test_rank_singular(self):
        p = MiniAPLParser()
        p.set_variable('M', np.array([[1.0, 2.0], [2.0, 4.0]]))
        result = p.evaluate("rank(M)")
        assert result == 1

    def test_sort_default(self):
        p = MiniAPLParser()
        p.set_variable('v', np.array([3.0, 1.0, 2.0]))
        result = p.evaluate("sort(v)")
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)

    def test_dimension_mismatch_add(self):
        p = MiniAPLParser()
        p.set_variables(a=np.ones((3, 4)), b=np.ones((5, 6)))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            p.evaluate("a + b")

    def test_dimension_mismatch_mul(self):
        p = MiniAPLParser()
        p.set_variables(a=np.ones((10,)), b=np.ones((20, 30)))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            p.evaluate("a x b")


class TestNewPrimitives:
    def test_rank_matrix(self):
        p = MiniAPLParser()
        p.set_variable('M', np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = p.evaluate("rank(M)")
        assert result == 2

    def test_rank_singular(self):
        p = MiniAPLParser()
        p.set_variable('M', np.array([[1.0, 2.0], [2.0, 4.0]]))
        result = p.evaluate("rank(M)")
        assert result == 1

    def test_sort_default(self):
        p = MiniAPLParser()
        p.set_variable('v', np.array([3.0, 1.0, 2.0]))
        result = p.evaluate("sort(v)")
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(result, expected)

    def test_dimension_mismatch_add(self):
        p = MiniAPLParser()
        p.set_variables(a=np.ones((3, 4)), b=np.ones((5, 6)))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            p.evaluate("a + b")

    def test_dimension_mismatch_mul(self):
        p = MiniAPLParser()
        p.set_variables(a=np.ones((10,)), b=np.ones((20, 30)))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            p.evaluate("a x b")
