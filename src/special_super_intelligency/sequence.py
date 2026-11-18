"""Byte-level next-token data preparation and tiny training baseline."""

from __future__ import annotations

from .model import TinyMLP


def build_next_byte_samples(
    token_sequences: list[list[int]], context_length: int
) -> list[tuple[list[float], list[float]]]:
    """Create normalized sliding-window inputs and next-byte regression targets."""
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    samples: list[tuple[list[float], list[float]]] = []
    for tokens in token_sequences:
        if any(type(token) is not int or not 0 <= token <= 255 for token in tokens):
            raise ValueError("token sequences must contain byte IDs in the range 0..255.")
        for index in range(len(tokens)):
            context = tokens[max(0, index - context_length):index]
            padded = [0] * (context_length - len(context)) + context
            samples.append(([value / 255.0 for value in padded], [tokens[index] / 255.0]))
    if not samples:
        raise ValueError("At least one token is required to build training samples.")
    return samples


def train_next_byte_model(
    token_sequences: list[list[int]],
    *,
    context_length: int = 8,
    hidden_size: int = 16,
    epochs: int = 300,
    learning_rate: float = 0.05,
    seed: int = 7,
) -> tuple[TinyMLP, float, float]:
    """Train a small numeric next-byte baseline; this is not a general LLM."""
    samples = build_next_byte_samples(token_sequences, context_length)
    model = TinyMLP(context_length, hidden_size, 1, seed=seed)
    report = model.train(samples, epochs=epochs, learning_rate=learning_rate)
    return model, report.initial_loss, report.final_loss


def predict_next_byte(model: TinyMLP, context: list[int]) -> int:
    """Predict one byte ID from recent context using the trained regression model."""
    if len(context) > model.input_size:
        context = context[-model.input_size:]
    if any(type(token) is not int or not 0 <= token <= 255 for token in context):
        raise ValueError("context must contain byte IDs in the range 0..255.")
    padded = [0] * (model.input_size - len(context)) + context
    prediction = model.predict([value / 255.0 for value in padded])[0]
    return max(0, min(255, round(prediction * 255.0)))
