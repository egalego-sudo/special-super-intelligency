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
    before_query = [row[:] for row in model.blocks[0].query]
    before_ffn = [row[:] for row in model.blocks[0].ffn_in]
    initial, final = model.train(
        [([1, 2], 3), ([1, 2], 3), ([2, 1], 4)],
        epochs=8,
        learning_rate=0.015,
        max_grad_norm=1.0,
    )
    assert final < initial
    assert model.embedding[1][0] != before_embedding
    assert any(
        model.blocks[0].query[i][j] != before_query[i][j]
        for i in range(2) for j in range(2)
    )
    assert any(
        model.blocks[0].ffn_in[i][j] != before_ffn[i][j]
        for i in range(2) for j in range(8)
    )


def test_full_decoder_training_rejects_non_finite_gradients_configuration():
    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1)
    with pytest.raises(ValueError):
        model.train([([1], 2)], max_grad_norm=float("inf"))


def test_full_decoder_gradient_matches_finite_difference():
    import math

    from special_super_intelligency.transformer import DecoderLanguageModel

    model = DecoderLanguageModel(vocab_size=5, width=2, heads=1, layers=1, seed=79)
    samples = [([1, 2], 3)]
    parameter = model.blocks[0].ffn_in[0][0]
    epsilon = 1e-5

    def loss():
        logits = model.forward([1, 2])[-1]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        return -math.log(exps[3] / sum(exps))

    original = parameter
    model.blocks[0].ffn_in[0][0] = original + epsilon
    high = loss()
    model.blocks[0].ffn_in[0][0] = original - epsilon
    low = loss()
    model.blocks[0].ffn_in[0][0] = original
    numeric_gradient = (high - low) / (2 * epsilon)

    model.train(samples, epochs=1, learning_rate=1e-5, max_grad_norm=1e6, weight_decay=0.0, optimizer="sgd")
    learned_gradient = (original - model.blocks[0].ffn_in[0][0]) / 1e-5
    assert learned_gradient == pytest.approx(numeric_gradient, rel=2e-3, abs=2e-4)


def test_sampling_filters_top_k_and_normalizes_probabilities():
    model = DecoderLanguageModel(vocab_size=12, width=4, heads=2, layers=1, seed=139)
    logits = model.forward([1, 2])[-1]
    probabilities = model._sampling_probabilities(logits, temperature=0.8, top_k=3)
    assert sum(probabilities) == pytest.approx(1.0)
    assert sum(value > 0 for value in probabilities) <= 3


def test_nucleus_sampling_keeps_minimum_probability_mass():
    model = DecoderLanguageModel(vocab_size=10, width=4, heads=2, layers=1, seed=149)
    logits = model.forward([1, 2])[-1]
    probabilities = model._sampling_probabilities(logits, top_p=0.7)
    assert sum(probabilities) == pytest.approx(1.0)
    assert 1 <= sum(value > 0 for value in probabilities) <= 10


def test_seeded_sampling_and_generation_are_reproducible():
    model = DecoderLanguageModel(vocab_size=12, width=4, heads=2, layers=1, seed=151)
    first = model.sample_next_token([1, 2], temperature=0.9, top_k=5, seed=17)
    second = model.sample_next_token([1, 2], temperature=0.9, top_k=5, seed=17)
    assert first == second
    generated_a = model.generate([1, 2], max_new_tokens=5, temperature=0.9, top_p=0.9, seed=19)
    generated_b = model.generate([1, 2], max_new_tokens=5, temperature=0.9, top_p=0.9, seed=19)
    assert generated_a == generated_b
    assert len(generated_a) == 7


def test_default_generation_remains_greedy():
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=1, seed=157)
    expected = max(range(8), key=lambda index: model.next_token_probabilities([1])[index])
    assert model.generate([1], max_new_tokens=1)[-1] == expected


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0.0},
    {"temperature": float("nan")},
    {"top_k": 0},
    {"top_k": 99},
    {"top_p": 0.0},
    {"top_p": 1.1},
])
def test_sampling_rejects_invalid_parameters(kwargs):
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=1)
    with pytest.raises(ValueError):
        model.sample_next_token([1, 2], **kwargs)


def test_kv_cached_step_logits_match_full_sequence_forward():
    model = DecoderLanguageModel(vocab_size=16, width=4, heads=2, layers=2, seed=163)
    cache = model.create_kv_cache()
    tokens = [1, 2, 3, 4]
    for index, token in enumerate(tokens):
        cached_logits = model.forward_step(token, cache)
        full_logits = model.forward(tokens[:index + 1])[-1]
        assert cached_logits == pytest.approx(full_logits)
        assert cache.length == index + 1
        assert all(len(layer) == cache.length for layer in cache.keys)
        assert all(len(layer) == cache.length for layer in cache.values)


def test_kv_cache_rejects_overflow_and_inconsistent_state():
    model = DecoderLanguageModel(
        vocab_size=8, width=4, heads=2, layers=1, max_sequence_length=2
    )
    cache = model.create_kv_cache()
    model.forward_step(1, cache)
    model.forward_step(2, cache)
    with pytest.raises(ValueError, match="full"):
        model.forward_step(3, cache)

    broken = model.create_kv_cache()
    broken.keys[0].append([0.0] * 4)
    with pytest.raises(ValueError, match="inconsistent"):
        model.forward_step(1, broken)


def test_kv_cached_greedy_generation_matches_full_recompute_reference():
    model = DecoderLanguageModel(
        vocab_size=10, width=4, heads=2, layers=2, max_sequence_length=3, seed=167
    )
    prompt = [1, 2, 3, 4]
    expected = list(prompt)
    for _ in range(6):
        context = expected[-model.max_sequence_length:]
        logits = model.forward(context)[-1]
        expected.append(max(range(model.vocab_size), key=lambda index: logits[index]))
    assert model.generate(prompt, max_new_tokens=6) == expected


def test_kv_cache_supports_longer_generation_by_rolling_context():
    model = DecoderLanguageModel(
        vocab_size=9, width=4, heads=2, layers=1, max_sequence_length=2, seed=173
    )
    generated = model.generate([1], max_new_tokens=8)
    assert len(generated) == 9
    assert all(0 <= token < 9 for token in generated)
