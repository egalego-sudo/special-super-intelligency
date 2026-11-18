"""Small deterministic evaluation helpers for regression baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import TinyMLP


@dataclass(frozen=True)
class RegressionMetrics:
    examples: int
    mean_squared_error: float
    within_tolerance: int


def evaluate_regression(
    model: TinyMLP,
    samples: list[tuple[list[float], list[float]]],
    *,
    tolerance: float = 0.1,
) -> RegressionMetrics:
    """Evaluate predictions on held-out samples without updating model weights."""
    if not samples:
        raise ValueError("Evaluation requires at least one sample.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative.")
    squared_error = 0.0
    within_tolerance = 0
    count = 0
    for inputs, targets in samples:
        predictions = model.predict(inputs)
        if len(targets) != model.output_size:
            raise ValueError("Each target must match the model output size.")
        for prediction, target in zip(predictions, targets):
            if not math.isfinite(float(target)):
                raise ValueError("Targets must contain only finite values.")
            error = abs(prediction - target)
            squared_error += error * error
            within_tolerance += error <= tolerance
            count += 1
    return RegressionMetrics(count, squared_error / count, within_tolerance)
