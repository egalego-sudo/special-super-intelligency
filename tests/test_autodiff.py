import math

import pytest

from special_super_intelligency.autodiff import Value


def finite_difference(function, point, epsilon=1e-6):
    return (function(point + epsilon) - function(point - epsilon)) / (2 * epsilon)


def test_reverse_mode_matches_finite_difference_for_composite_expression():
    x = Value(1.7)
    y = Value(-0.4)
    result = (x * y + x**2).tanh() + (x / 2.0).exp()
    result.backward()

    def reference(value):
        return math.tanh(value * -0.4 + value**2) + math.exp(value / 2.0)

    assert x.grad == pytest.approx(finite_difference(reference, 1.7), rel=1e-5)
    expected_y = (1.0 - math.tanh(1.7 * -0.4 + 1.7**2) ** 2) * 1.7
    assert y.grad == pytest.approx(expected_y, rel=1e-6)


def test_shared_subexpressions_accumulate_gradients():
    x = Value(3.0)
    shared = x * x
    result = shared + shared * 2.0
    result.backward()
    assert x.grad == pytest.approx(18.0)


def test_backward_accumulates_across_calls_until_zeroed():
    x = Value(2.0)
    result = x**3
    result.backward()
    result.backward()
    assert x.grad == pytest.approx(24.0)
    result.zero_grad()
    assert x.grad == 0.0


def test_division_and_power_gradients():
    x = Value(4.0)
    y = Value(2.0)
    result = x / y + x**0.5
    result.backward()
    assert x.grad == pytest.approx(1 / 2 + 1 / 4)
    assert y.grad == pytest.approx(-1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_values_are_rejected(bad):
    with pytest.raises(ValueError):
        Value(bad)
