import pytest

from special_super_intelligency.batching import (
    pad_next_token_samples,
    pad_sequences,
)
from special_super_intelligency.transformer import DecoderLanguageModel


def test_pad_sequences_builds_right_padding_and_attention_masks():
    batch = pad_sequences([[1, 2], [3, 4, 5]], pad_token_id=0)
    assert batch.input_ids == [[1, 2, 0], [3, 4, 5]]
    assert batch.attention_mask == [[True, True, False], [True, True, True]]
    assert batch.lengths == [2, 3]


def test_pad_next_token_samples_preserves_targets():
    padded = pad_next_token_samples([([1], 2), ([1, 2, 3], 4)])
    assert padded == [
        ([1, 0, 0], [True, False, False], 2),
        ([1, 2, 3], [True, True, True], 4),
    ]


def test_padded_batch_logits_match_unpadded_sequences():
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=2, seed=101)
    batch = pad_sequences([[1, 2], [3, 4, 5]], pad_token_id=0)
    outputs = model.forward_batch(batch)
    expected = [model.forward([1, 2]), model.forward([3, 4, 5])]
    for actual_sequence, expected_sequence in zip(outputs, expected):
        for actual, reference in zip(actual_sequence, expected_sequence):
            assert actual == pytest.approx(reference)


def test_padding_token_values_cannot_leak_into_valid_logits():
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=2, seed=103)
    first = model.forward([1, 2, 0, 0], attention_mask=[True, True, False, False])
    second = model.forward([1, 2, 7, 6], attention_mask=[True, True, False, False])
    assert first[:2] == second[:2]
    assert first[:2] == model.forward([1, 2])


def test_non_right_padded_attention_mask_is_rejected():
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=1)
    with pytest.raises(ValueError, match="True prefix"):
        model.forward([1, 2, 3], attention_mask=[True, False, True])


def test_model_trains_on_masked_minibatches():
    model = DecoderLanguageModel(vocab_size=8, width=2, heads=1, layers=1, seed=107)
    samples = pad_next_token_samples([([1], 2), ([1, 2], 3), ([2], 3)])
    initial, final = model.train(
        samples, epochs=4, batch_size=2, learning_rate=0.01, max_grad_norm=1.0
    )
    assert final < initial


def test_training_rejects_invalid_minibatch_size():
    model = DecoderLanguageModel(vocab_size=8, width=2, heads=1, layers=1)
    with pytest.raises(ValueError, match="batch_size"):
        model.train([([1], 2)], batch_size=0)
