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


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    model = TinyMLP(2, 3, 1, seed=19)
    model.train([([0.0, 0.0], [0.0]), ([1.0, 1.0], [1.0])], epochs=10)
    checkpoint = tmp_path / "model.json"
    model.save_checkpoint(str(checkpoint))
    restored = TinyMLP.load_checkpoint(str(checkpoint))
    assert restored.predict([0.3, 0.7]) == model.predict([0.3, 0.7])


def test_checkpoint_rejects_unknown_version(tmp_path):
    checkpoint = tmp_path / "bad.json"
    checkpoint.write_text('{"format_version": 999}', encoding="utf-8")
    with pytest.raises(ValueError):
        TinyMLP.load_checkpoint(str(checkpoint))


def test_backprop_gradient_matches_finite_difference(tmp_path):
    model = TinyMLP(2, 3, 1, seed=23)
    checkpoint = tmp_path / "before.json"
    model.save_checkpoint(str(checkpoint))
    updated = TinyMLP.load_checkpoint(str(checkpoint))
    sample_x = [0.2, -0.4]
    sample_y = [0.7]
    learning_rate = 1e-5
    original_weight = model.w1[0][0]

    def sample_loss(candidate):
        model.w1[0][0] = candidate
        prediction = model.predict(sample_x)[0]
        return (prediction - sample_y[0]) ** 2

    epsilon = 1e-6
    numeric_gradient = (
        sample_loss(original_weight + epsilon)
        - sample_loss(original_weight - epsilon)
    ) / (2 * epsilon)
    model.w1[0][0] = original_weight

    updated.train([(sample_x, sample_y)], epochs=1, learning_rate=learning_rate)
    analytic_gradient = (original_weight - updated.w1[0][0]) / learning_rate
    assert analytic_gradient == pytest.approx(numeric_gradient, rel=2e-3, abs=2e-5)


def test_rejects_non_integer_layer_dimensions():
    with pytest.raises(ValueError):
        TinyMLP(2.5, 4, 1)


def test_rejects_non_numeric_inputs():
    with pytest.raises(TypeError):
        TinyMLP(1, 2, 1).predict(["not-a-number"])  # type: ignore[list-item]


def test_rejects_non_integer_epoch_count():
    with pytest.raises(ValueError):
        TinyMLP(1, 2, 1).train([([0.0], [0.0])], epochs=1.5)  # type: ignore[arg-type]


def test_checkpoint_rejects_invalid_dimensions(tmp_path):
    checkpoint = tmp_path / "bad-dimensions.json"
    checkpoint.write_text(
        '{"format_version":1,"input_size":0,"hidden_size":2,"output_size":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        TinyMLP.load_checkpoint(str(checkpoint))
