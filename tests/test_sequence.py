import pytest

from special_super_intelligency.sequence import (
    build_next_byte_samples,
    predict_next_byte,
    train_next_byte_model,
)


def test_builds_sliding_window_samples():
    samples = build_next_byte_samples([[10, 20, 30]], context_length=2)
    assert samples == [
        ([0.0, 0.0], [10 / 255.0]),
        ([0.0, 10 / 255.0], [20 / 255.0]),
        ([10 / 255.0, 20 / 255.0], [30 / 255.0]),
    ]


def test_rejects_invalid_context_length():
    with pytest.raises(ValueError):
        build_next_byte_samples([[1, 2]], context_length=0)


def test_rejects_empty_sequences():
    with pytest.raises(ValueError):
        build_next_byte_samples([[]], context_length=2)


def test_training_reduces_toy_corpus_loss():
    corpus = [list(b"abcabcabcabcabc")]
    model, initial_loss, final_loss = train_next_byte_model(
        corpus, context_length=2, hidden_size=8, epochs=500, learning_rate=0.08
    )
    assert final_loss < initial_loss
    prediction = predict_next_byte(model, list(b"ab"))
    assert 0 <= prediction <= 255


def test_predict_rejects_invalid_token():
    _, _, _ = train_next_byte_model([[1, 2, 3]], context_length=2, epochs=1)
    from special_super_intelligency.model import TinyMLP
    with pytest.raises(ValueError):
        predict_next_byte(TinyMLP(2, 3, 1), [999])
