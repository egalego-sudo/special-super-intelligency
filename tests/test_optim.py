import pytest

from special_super_intelligency.autodiff import Value
from special_super_intelligency.optim import AdamW


def test_adamw_reduces_quadratic_objective():
    parameter = Value(4.0)
    optimizer = AdamW(learning_rate=0.08, weight_decay=0.0)
    initial = parameter.data**2
    for _ in range(100):
        parameter.grad = 0.0
        (parameter * parameter).backward()
        optimizer.step([parameter])
    assert parameter.data**2 < initial * 0.01
    assert optimizer.steps == 100


def test_adamw_clips_global_gradient_norm():
    first, second = Value(1.0), Value(2.0)
    first.grad, second.grad = 30.0, 40.0
    optimizer = AdamW(learning_rate=0.01, weight_decay=0.0, max_grad_norm=1.0)
    norm, scale = optimizer.step([first, second])
    assert norm == pytest.approx(50.0)
    assert scale == pytest.approx(0.02)


def test_adamw_applies_decoupled_weight_decay():
    parameter = Value(2.0)
    parameter.grad = 0.0
    optimizer = AdamW(learning_rate=0.1, weight_decay=0.5)
    optimizer.step([parameter])
    assert parameter.data == pytest.approx(1.9)


@pytest.mark.parametrize("kwargs", [
    {"learning_rate": 0.0},
    {"betas": (1.0, 0.9)},
    {"epsilon": 0.0},
    {"weight_decay": -0.1},
    {"max_grad_norm": float("inf")},
])
def test_adamw_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        AdamW(**kwargs)


def test_adamw_rejects_non_finite_gradients():
    parameter = Value(1.0)
    parameter.grad = float("nan")
    with pytest.raises(ArithmeticError):
        AdamW().step([parameter])
