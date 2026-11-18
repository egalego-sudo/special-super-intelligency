"""A tiny, dependency-free neural network research baseline.

This educational MLP demonstrates trainable independent weights and inference.
It is not a language model, Transformer, or production foundation model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingReport:
    initial_loss: float
    final_loss: float
    epochs: int


class TinyMLP:
    """One-hidden-layer tanh network trained with batch gradient descent."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 7):
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("All layer sizes must be positive.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        rng = random.Random(seed)
        scale1 = 1.0 / math.sqrt(input_size)
        scale2 = 1.0 / math.sqrt(hidden_size)
        self.w1 = [[rng.uniform(-scale1, scale1) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[rng.uniform(-scale2, scale2) for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.b2 = [0.0] * output_size

    @staticmethod
    def _validate_vector(values: list[float], expected: int, name: str) -> None:
        if len(values) != expected:
            raise ValueError(f"{name} must contain exactly {expected} values.")
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} must contain only finite numbers.")

    def predict(self, inputs: list[float]) -> list[float]:
        self._validate_vector(inputs, self.input_size, "inputs")
        hidden = [
            math.tanh(bias + sum(weight * value for weight, value in zip(row, inputs)))
            for row, bias in zip(self.w1, self.b1)
        ]
        return [
            bias + sum(weight * value for weight, value in zip(row, hidden))
            for row, bias in zip(self.w2, self.b2)
        ]

    def train(
        self,
        samples: list[tuple[list[float], list[float]]],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
    ) -> TrainingReport:
        if not samples:
            raise ValueError("At least one training sample is required.")
        if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive.")
        for inputs, targets in samples:
            self._validate_vector(inputs, self.input_size, "inputs")
            self._validate_vector(targets, self.output_size, "targets")

        def loss() -> float:
            total = 0.0
            for inputs, targets in samples:
                predictions = self.predict(inputs)
                total += sum((p - t) ** 2 for p, t in zip(predictions, targets))
            return total / (len(samples) * self.output_size)

        initial_loss = loss()
        for _ in range(epochs):
            grad_w1 = [[0.0] * self.input_size for _ in range(self.hidden_size)]
            grad_b1 = [0.0] * self.hidden_size
            grad_w2 = [[0.0] * self.hidden_size for _ in range(self.output_size)]
            grad_b2 = [0.0] * self.output_size

            for inputs, targets in samples:
                hidden = [
                    math.tanh(bias + sum(weight * value for weight, value in zip(row, inputs)))
                    for row, bias in zip(self.w1, self.b1)
                ]
                predictions = [
                    bias + sum(weight * value for weight, value in zip(row, hidden))
                    for row, bias in zip(self.w2, self.b2)
                ]
                delta2 = [
                    2.0 * (prediction - target) / (len(samples) * self.output_size)
                    for prediction, target in zip(predictions, targets)
                ]
                delta1 = [
                    (1.0 - hidden[j] ** 2)
                    * sum(self.w2[k][j] * delta2[k] for k in range(self.output_size))
                    for j in range(self.hidden_size)
                ]
                for k, delta in enumerate(delta2):
                    grad_b2[k] += delta
                    for j, value in enumerate(hidden):
                        grad_w2[k][j] += delta * value
                for j, delta in enumerate(delta1):
                    grad_b1[j] += delta
                    for i, value in enumerate(inputs):
                        grad_w1[j][i] += delta * value

            for j in range(self.hidden_size):
                self.b1[j] -= learning_rate * grad_b1[j]
                for i in range(self.input_size):
                    self.w1[j][i] -= learning_rate * grad_w1[j][i]
            for k in range(self.output_size):
                self.b2[k] -= learning_rate * grad_b2[k]
                for j in range(self.hidden_size):
                    self.w2[k][j] -= learning_rate * grad_w2[k][j]

        return TrainingReport(initial_loss, loss(), epochs)
