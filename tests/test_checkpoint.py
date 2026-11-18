import json

import pytest

from special_super_intelligency.transformer import DecoderLanguageModel


def test_checkpoint_round_trip_preserves_model_outputs(tmp_path):
    model = DecoderLanguageModel(
        vocab_size=13, width=4, heads=2, layers=2, max_sequence_length=16, seed=83
    )
    model.train(
        [([1, 2], 3), ([2, 3], 4)],
        epochs=3,
        learning_rate=0.01,
        max_grad_norm=1.0,
    )
    expected = model.forward([1, 2, 3])
    path = tmp_path / "quantum-ai.json"
    model.save_checkpoint(path)

    restored = DecoderLanguageModel.load_checkpoint(path)
    assert restored.forward([1, 2, 3]) == pytest.approx(expected)
    assert restored.vocab_size == model.vocab_size
    assert len(restored.blocks) == len(model.blocks)


def test_checkpoint_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad-version.json"
    path.write_text(json.dumps({
        "format_version": 999,
        "model": {},
        "state": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        DecoderLanguageModel.load_checkpoint(path)


def test_checkpoint_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        DecoderLanguageModel.load_checkpoint(path)


def test_checkpoint_rejects_non_finite_weights(tmp_path):
    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1)
    path = tmp_path / "nan.json"
    model.save_checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["embedding"][0][0] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        DecoderLanguageModel.load_checkpoint(path)


def test_checkpoint_rejects_wrong_parameter_shape(tmp_path):
    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1)
    path = tmp_path / "shape.json"
    model.save_checkpoint(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["state"]["output_projection"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        DecoderLanguageModel.load_checkpoint(path)
