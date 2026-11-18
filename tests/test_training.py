import math

import pytest

from special_super_intelligency.dataset import TextExample
from special_super_intelligency.tokenizer import ByteTokenizer
from special_super_intelligency.training import (
    build_next_token_samples,
    evaluate_examples,
    scheduled_learning_rate,
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
        examples, approved_licenses=("CC0",), epochs=2, learning_rate=0.01, max_grad_norm=1.0
    )
    metrics = model.evaluate_text(examples, approved_licenses=("CC0",))
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
        model.train_text(examples, approved_licenses=("CC0",))


def test_evaluation_rejects_empty_text_pairs():
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1)
    with pytest.raises(ValueError, match="no next-token"):
        evaluate_examples(model, [TextExample("x", "source", "CC0")], approved_licenses=("CC0",))

def test_fit_selects_and_restores_best_validation_checkpoint(tmp_path):
    training = [
        TextExample("abab", "author training fixture", "CC0"),
        TextExample("baba", "author training fixture", "CC0"),
    ]
    validation = [TextExample("abab", "separate validation author", "CC0")]
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1, seed=109)
    checkpoint = tmp_path / "best-model.json"

    result = model.fit(
        training,
        validation,
        epochs=3,
        batch_size=2,
        learning_rate=0.005,
        seed=17,
        checkpoint_path=checkpoint,
        approved_licenses=("CC0",),
    )
    assert result["best_epoch"] in (1, 2, 3)
    assert len(result["history"]) == 3
    assert result["training_samples"] == 6
    assert result["validation_samples"] == 3
    assert checkpoint.exists()

    restored = DecoderLanguageModel.load_checkpoint(checkpoint)
    actual = model.forward([97, 98])
    expected = restored.forward([97, 98])
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row)


def test_fit_is_reproducible_for_same_seed():
    training = [TextExample("abab", "train author", "CC0")]
    validation = [TextExample("baba", "validation author", "CC0")]
    first = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1, seed=113)
    second = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1, seed=113)

    first_result = first.fit(training, validation, epochs=2, seed=23, learning_rate=0.005, approved_licenses=("CC0",))
    second_result = second.fit(training, validation, epochs=2, seed=23, learning_rate=0.005, approved_licenses=("CC0",))
    assert first_result == second_result
    assert first.forward([97, 98]) == second.forward([97, 98])


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("constant", [0.1, 0.1, 0.1, 0.1]),
        ("linear", [0.1, 0.06666666666666667, 0.03333333333333334, 0.0]),
        ("cosine", [0.1, 0.075, 0.025, 0.0]),
    ],
)
def test_learning_rate_schedules(schedule, expected):
    actual = [
        scheduled_learning_rate(0.1, epoch, 4, schedule=schedule, minimum_rate=0.0)
        for epoch in range(1, 5)
    ]
    assert actual == pytest.approx(expected)


def test_learning_rate_warmup_increases_then_decays():
    rates = [
        scheduled_learning_rate(
            0.1, epoch, 5, schedule="linear", warmup_epochs=2, minimum_rate=0.01
        )
        for epoch in range(1, 6)
    ]
    assert rates[0] < rates[1] == pytest.approx(0.1)
    assert rates[2] > rates[3] > rates[4]


def test_fit_early_stops_after_patience_and_restores_best(tmp_path):
    training = [TextExample("abab", "training author", "CC0")]
    validation = [TextExample("baba", "validation author", "CC0")]
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1, seed=127)
    result = model.fit(
        training,
        validation,
        epochs=8,
        learning_rate=0.005,
        patience=1,
        min_delta=100.0,
        checkpoint_path=tmp_path / "early-stop.json",
        approved_licenses=("CC0",),
    )
    assert result["stopped_early"] is True
    assert result["epochs_completed"] == 2
    assert result["best_epoch"] == 1
    assert len(result["history"]) == 2


def test_fit_rejects_invalid_early_stopping_configuration():
    training = [TextExample("abab", "training author", "CC0")]
    validation = [TextExample("baba", "validation author", "CC0")]
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1)
    with pytest.raises(ValueError, match="patience"):
        model.fit(training, validation, epochs=2, patience=0, approved_licenses=("CC0",))


def test_tiny_corpus_experiment_measures_overfit_and_heldout_generalization():
    import math

    from special_super_intelligency.experiment import run_decoder_tiny_corpus_experiment

    report = run_decoder_tiny_corpus_experiment(epochs=10, seed=131)
    assert report["synthetic_only"] is True
    assert report["training_loss_after"] < report["training_loss_before"]
    assert math.isfinite(report["validation_loss_before"])
    assert math.isfinite(report["validation_loss_after"])
    assert math.isfinite(report["validation_perplexity"])
    assert 0.0 <= report["validation_accuracy"] <= 1.0
    assert math.isfinite(report["generalization_gap"])
    assert report["training_tokens"] == 3
    assert report["validation_tokens"] == 3


def test_provenance_audit_returns_source_license_counts_and_hash():
    from special_super_intelligency.training import audit_provenance

    examples = [
        TextExample("alpha", "author A", "CC0"),
        TextExample("beta", "author A", "CC0"),
        TextExample("gamma", "author B", "CC0"),
    ]
    report = audit_provenance(examples, approved_licenses=("CC0",))
    assert report["records"] == 3
    assert report["license_counts"] == {"CC0": 3}
    assert report["source_counts"] == {"author A": 2, "author B": 1}
    assert len(report["dataset_sha256"]) == 64
    assert report["dataset_sha256"] == audit_provenance(
        examples, approved_licenses=("CC0",)
    )["dataset_sha256"]


def test_provenance_audit_rejects_unapproved_license():
    from special_super_intelligency.training import audit_provenance

    examples = [TextExample("text", "unknown source", "All Rights Reserved")]
    with pytest.raises(ValueError, match="unapproved license"):
        audit_provenance(examples, approved_licenses=("CC0",))


def test_provenance_audit_requires_explicit_allowlist():
    from special_super_intelligency.training import audit_provenance

    examples = [TextExample("text", "author", "CC0")]
    with pytest.raises(ValueError, match="approved_licenses"):
        audit_provenance(examples, approved_licenses=None)


def test_fit_rejects_unapproved_validation_license():
    training = [TextExample("abab", "training author", "CC0")]
    validation = [TextExample("baba", "validation author", "All Rights Reserved")]
    model = DecoderLanguageModel(vocab_size=256, width=2, heads=1, layers=1)
    with pytest.raises(ValueError, match="unapproved license"):
        model.fit(
            training,
            validation,
            epochs=1,
            approved_licenses=("CC0",),
        )
