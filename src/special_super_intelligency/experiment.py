"""Reproducible experiment configuration and execution entry point."""

from __future__ import annotations

from dataclasses import dataclass

from .model import TinyMLP
from .sequence import build_next_byte_samples, train_next_byte_model


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
