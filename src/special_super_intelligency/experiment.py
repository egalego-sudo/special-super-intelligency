"""Reproducible experiment configuration and execution entry point."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .model import TinyMLP
from .sequence import train_next_byte_model


@dataclass(frozen=True)
class ExperimentConfig:
    context_length: int = 8
    hidden_size: int = 16
    epochs: int = 300
    learning_rate: float = 0.05
    seed: int = 7

    def validate(self) -> None:
        if self.context_length <= 0 or self.hidden_size <= 0 or self.epochs <= 0:
            raise ValueError("context_length, hidden_size, and epochs must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


def run_experiment(
    token_sequences: list[list[int]], config: ExperimentConfig
) -> tuple[TinyMLP, float, float]:
    """Run a deterministic next-byte experiment from an explicit config."""
    config.validate()
    return train_next_byte_model(
        token_sequences,
        context_length=config.context_length,
        hidden_size=config.hidden_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )


def run_decoder_tiny_corpus_experiment(*, epochs=20, seed=131):
    """Train a tiny authored corpus and report held-out generalization metrics."""
    from .dataset import TextExample
    from .training import evaluate_examples
    from .transformer import DecoderLanguageModel

    training_examples = [
        TextExample("abab", "QUANTUM AI synthetic training fixture", "CC0")
    ]
    validation_examples = [
        TextExample("baba", "QUANTUM AI synthetic held-out fixture", "CC0")
    ]
    model = DecoderLanguageModel(
        vocab_size=256, width=2, heads=1, layers=1,
        max_sequence_length=16, seed=seed,
    )
    before_training = evaluate_examples(model, training_examples, approved_licenses=("CC0",))
    before_validation = evaluate_examples(model, validation_examples, approved_licenses=("CC0",))
    fit = model.fit(
        training_examples,
        validation_examples,
        epochs=epochs,
        batch_size=2,
        learning_rate=0.01,
        max_grad_norm=1.0,
        weight_decay=0.0,
        schedule="cosine",
        patience=5,
        seed=seed,
        approved_licenses=("CC0",),
    )
    after_training = evaluate_examples(model, training_examples, approved_licenses=("CC0",))
    after_validation = evaluate_examples(model, validation_examples, approved_licenses=("CC0",))
    return {
        "experiment": "tiny-corpus-overfit-heldout-generalization-v1",
        "synthetic_only": True,
        "seed": seed,
        "training_text": training_examples[0].text,
        "validation_text": validation_examples[0].text,
        "epochs_requested": epochs,
        "epochs_completed": fit["epochs_completed"],
        "best_epoch": fit["best_epoch"],
        "training_loss_before": before_training["loss"],
        "training_loss_after": after_training["loss"],
        "validation_loss_before": before_validation["loss"],
        "validation_loss_after": after_validation["loss"],
        "validation_perplexity": after_validation["perplexity"],
        "validation_accuracy": after_validation["accuracy"],
        "generalization_gap": after_validation["loss"] - after_training["loss"],
        "training_tokens": after_training["tokens"],
        "validation_tokens": after_validation["tokens"],
    }


def run_decoder_seed_sweep(*, seeds=(131, 137, 139), epochs=20, output_path=None):
    """Run identical synthetic decoder experiments across seeds and hash the report."""
    from .dataset import TextExample
    from .training import audit_provenance

    if (not isinstance(seeds, (list, tuple)) or len(seeds) < 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must contain at least two unique non-negative integers.")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")

    training_examples = [
        TextExample("abab", "QUANTUM AI synthetic training fixture", "CC0")
    ]
    validation_examples = [
        TextExample("baba", "QUANTUM AI synthetic held-out fixture", "CC0")
    ]
    training_provenance = audit_provenance(
        training_examples, approved_licenses=("CC0",)
    )
    validation_provenance = audit_provenance(
        validation_examples, approved_licenses=("CC0",)
    )
    runs = [
        run_decoder_tiny_corpus_experiment(epochs=epochs, seed=seed)
        for seed in seeds
    ]
    metric_names = (
        "training_loss_after",
        "validation_loss_after",
        "validation_perplexity",
        "validation_accuracy",
        "generalization_gap",
    )
    aggregate = {}
    for metric in metric_names:
        values = [run[metric] for run in runs]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "sample_stddev": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite experiment metric: {metric}.")

    payload = {
        "experiment": "decoder-multi-seed-sweep-v1",
        "synthetic_only": True,
        "epochs_requested": epochs,
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "training_provenance": training_provenance,
        "validation_provenance": validation_provenance,
        "runs": runs,
        "aggregate": aggregate,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    report = {**payload, "report_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}

    if output_path is not None:
        destination = Path(output_path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(report, temporary, sort_keys=True, indent=2, allow_nan=False)
                temporary.write("\\n")
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    return report
