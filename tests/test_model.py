import pytest

from special_super_intelligency.model import TinyMLP


def test_mlp_learns_simple_identity_mapping():
    model = TinyMLP(input_size=2, hidden_size=8, output_size=2, seed=3)
    samples = [
        ([0.0, 0.0], [0.0, 0.0]),
        ([1.0, 0.0], [1.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0]),
        ([1.0, 1.0], [1.0, 1.0]),
    ]
    report = model.train(samples, epochs=1200, learning_rate=0.08)
    assert report.final_loss < report.initial_loss
    assert report.final_loss < 0.02


def test_same_seed_produces_same_initial_predictions():
    first = TinyMLP(2, 4, 1, seed=11)
    second = TinyMLP(2, 4, 1, seed=11)
    assert first.predict([0.25, 0.75]) == second.predict([0.25, 0.75])


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        TinyMLP(0, 4, 1)


def test_rejects_wrong_input_shape():
    with pytest.raises(ValueError):
        TinyMLP(2, 4, 1).predict([1.0])


def test_rejects_non_finite_input():
    with pytest.raises(ValueError):
        TinyMLP(1, 2, 1).predict([float("nan")])


def test_rejects_empty_training_data():
    with pytest.raises(ValueError):
        TinyMLP(1, 2, 1).train([])
