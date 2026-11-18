import pytest

from special_super_intelligency.evaluation import evaluate_regression
from special_super_intelligency.model import TinyMLP


def test_evaluation_reports_mse_without_training():
    model = TinyMLP(1, 2, 1, seed=1)
    before = model.predict([0.5])
    metrics = evaluate_regression(model, [([0.5], [0.0])], tolerance=10.0)
    assert metrics.examples == 1
    assert metrics.mean_squared_error >= 0
    assert model.predict([0.5]) == before


def test_evaluation_rejects_empty_data():
    with pytest.raises(ValueError):
        evaluate_regression(TinyMLP(1, 2, 1), [])


def test_evaluation_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        evaluate_regression(TinyMLP(1, 2, 1), [([0.0], [0.0])], tolerance=-1)


def test_evaluation_rejects_target_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate_regression(TinyMLP(1, 2, 1), [([0.0], [0.0, 1.0])])
