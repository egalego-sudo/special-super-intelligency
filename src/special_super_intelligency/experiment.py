"""Reproducible experiment configuration and execution entry point."""

from __future__ import annotations

from dataclasses import dataclass

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
    before_training = evaluate_examples(model, training_examples)
    before_validation = evaluate_examples(model, validation_examples)
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
    )
    after_training = evaluate_examples(model, training_examples)
    after_validation = evaluate_examples(model, validation_examples)
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
