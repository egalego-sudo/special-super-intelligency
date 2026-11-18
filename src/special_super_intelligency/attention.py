"""Small causal self-attention research primitive (single head, pure Python)."""

from __future__ import annotations

import math


def _validate_matrix(matrix: list[list[float]], size: int, name: str) -> None:
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError(f"{name} must be a square {size}x{size} matrix.")
    if any(not math.isfinite(float(value)) for row in matrix for value in row):
        raise ValueError(f"{name} must contain only finite values.")


def causal_self_attention(
    sequence: list[list[float]],
    query_weights: list[list[float]],
    key_weights: list[list[float]],
    value_weights: list[list[float]],
) -> list[list[float]]:
    """Apply scaled dot-product attention while masking all future positions."""
    if not sequence:
        raise ValueError("sequence must contain at least one token.")
    width = len(sequence[0])
    if width <= 0 or any(len(token) != width for token in sequence):
        raise ValueError("sequence must be a non-empty rectangular matrix.")
    if any(not math.isfinite(float(value)) for token in sequence for value in token):
        raise ValueError("sequence must contain only finite values.")
    _validate_matrix(query_weights, width, "query_weights")
    _validate_matrix(key_weights, width, "key_weights")
    _validate_matrix(value_weights, width, "value_weights")

    def project(token: list[float], weights: list[list[float]]) -> list[float]:
        return [sum(token[i] * weights[i][j] for i in range(width))
                for j in range(width)]

    queries = [project(token, query_weights) for token in sequence]
    keys = [project(token, key_weights) for token in sequence]
    values = [project(token, value_weights) for token in sequence]
    scale = math.sqrt(width)
    outputs: list[list[float]] = []

    for position, query in enumerate(queries):
        scores = [
            sum(query[i] * keys[j][i] for i in range(width)) / scale
            for j in range(position + 1)
        ]
        max_score = max(scores)
        exponentials = [math.exp(score - max_score) for score in scores]
        denominator = sum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        outputs.append([
            sum(probabilities[j] * values[j][dimension]
                for j in range(position + 1))
            for dimension in range(width)
        ])
    return outputs
