import math

import pytest

from special_super_intelligency.dataset import TextExample
from special_super_intelligency.tokenizer import ByteTokenizer
from special_super_intelligency.training import (
    build_next_token_samples,
    evaluate_examples,
)
from special_super_intelligency.transformer import DecoderLanguageModel


def test_byte_dataset_builds_next_token_samples_with_context_window():
    examples = [TextExample("abcd", "author-created test", "CC0")]
    samples = build_next_token_samples(examples, ByteTokenizer(), 2)
    assert samples == [([97], 98), ([97, 98], 99), ([98, 99], 100)]


def test_model_trains_and_evaluates_provenance_tagged_text():
    examples = [
        TextExample("aba", "author-created fixture", "CC0"),
        TextExample("aba", "author-created fixture", "CC0"),
    ]
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1, seed=97)
    initial, final = model.train_text(
        examples, epochs=2, learning_rate=0.01, max_grad_norm=1.0
    )
    metrics = model.evaluate_text(examples)
    assert math.isfinite(initial)
    assert math.isfinite(final)
    assert set(metrics) == {"loss", "perplexity", "accuracy", "tokens"}
    assert metrics["tokens"] == 4
    assert metrics["perplexity"] >= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_training_pipeline_rejects_missing_provenance():
    examples = [TextExample("text", "", "CC0")]
    with pytest.raises(ValueError, match="source"):
        build_next_token_samples(examples, ByteTokenizer(), 8)


def test_training_pipeline_rejects_vocabulary_mismatch():
    model = DecoderLanguageModel(vocab_size=32, width=2, heads=1, layers=1)
    examples = [TextExample("abc", "test author", "CC0")]
    with pytest.raises(ValueError, match="vocabulary"):
        model.train_text(examples)


def test_evaluation_rejects_empty_text_pairs():
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1)
    with pytest.raises(ValueError, match="no next-token"):
        evaluate_examples(model, [TextExample("x", "source", "CC0")])
