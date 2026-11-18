import pytest

from special_super_intelligency.transformer import DecoderTransformerBlock


def test_decoder_block_preserves_sequence_shape():
    block = DecoderTransformerBlock(width=8, heads=2, seed=13)
    result = block.forward([[0.1] * 8, [0.2] * 8, [0.3] * 8])
    assert len(result) == 3
    assert all(len(token) == 8 for token in result)


def test_decoder_block_is_deterministic_for_same_seed():
    first = DecoderTransformerBlock(width=8, heads=2, seed=17)
    second = DecoderTransformerBlock(width=8, heads=2, seed=17)
    sample = [[float(i + j) / 10 for j in range(8)] for i in range(3)]
    assert first.forward(sample) == second.forward(sample)


def test_decoder_block_masks_future_tokens():
    block = DecoderTransformerBlock(width=4, heads=2, seed=19)
    original = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]]
    changed_future = [original[0], original[1], [-9.0, 8.0, -7.0, 6.0]]
    before = block.forward(original)
    after = block.forward(changed_future)
    assert before[0] == after[0]
    assert before[1] == after[1]


def test_decoder_block_rejects_non_divisible_heads():
    with pytest.raises(ValueError):
        DecoderTransformerBlock(width=7, heads=2)


def test_decoder_block_rejects_bad_sequence_width():
    block = DecoderTransformerBlock(width=4, heads=2)
    with pytest.raises(ValueError):
        block.forward([[1.0, 2.0], [3.0, 4.0]])


def test_decoder_block_rejects_non_finite_input():
    block = DecoderTransformerBlock(width=2, heads=1)
    with pytest.raises(ValueError):
        block.forward([[float("nan"), 0.0]])


def test_language_model_returns_vocabulary_logits():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=16, width=8, heads=2, layers=2, seed=23)
    logits = model.forward([1, 2, 3])
    assert len(logits) == 3
    assert all(len(position) == 16 for position in logits)


def test_language_model_probabilities_are_normalized():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=11, width=4, heads=2, layers=1)
    probabilities = model.next_token_probabilities([1, 2])
    assert len(probabilities) == 11
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)


def test_language_model_generation_is_autoregressive():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=9, width=4, heads=2, layers=1, seed=31)
    generated = model.generate([1, 2], max_new_tokens=3)
    assert len(generated) == 5
    assert generated[:2] == [1, 2]
    assert all(0 <= token < 9 for token in generated)


def test_language_model_rejects_out_of_range_tokens():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=4, heads=2, layers=1)
    with pytest.raises(ValueError):
        model.forward([5])


def test_language_model_rejects_excessive_context():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(
        vocab_size=5, width=4, heads=2, layers=1, max_sequence_length=2
    )
    with pytest.raises(ValueError):
        model.forward([1, 2, 3])


def test_language_model_does_not_leak_future_tokens():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=12, width=4, heads=2, layers=2, seed=41)
    first = model.forward([1, 2, 3])
    changed = model.forward([1, 2, 9])
    assert first[0] == changed[0]
    assert first[1] == changed[1]


def test_language_model_trains_output_head_and_reduces_loss():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=7, width=4, heads=2, layers=1, seed=53)
    samples = [([1, 2], 3), ([1, 2], 3), ([2, 1], 3)]
    initial, final = model.train_output_head(samples, epochs=80, learning_rate=0.2)
    assert final < initial


def test_language_model_training_rejects_invalid_target():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=4, heads=2, layers=1)
    with pytest.raises(ValueError):
        model.train_output_head([([1, 2], 5)])


def test_full_decoder_training_updates_embeddings_attention_and_ffn():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1, seed=67)
    before_embedding = model.embedding[1][0]
    before_query = model.blocks[0].query[0][0]
    before_ffn = model.blocks[0].ffn_in[0][0]
    initial, final = model.train(
        [([1], 2), ([1], 2), ([2], 3)],
        epochs=8,
        learning_rate=0.015,
        max_grad_norm=1.0,
    )
    assert final < initial
    assert model.embedding[1][0] != before_embedding
    assert model.blocks[0].query[0][0] != before_query
    assert model.blocks[0].ffn_in[0][0] != before_ffn


def test_full_decoder_training_rejects_non_finite_gradients_configuration():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1)
    with pytest.raises(ValueError):
        model.train([([1], 2)], max_grad_norm=float("inf"))
