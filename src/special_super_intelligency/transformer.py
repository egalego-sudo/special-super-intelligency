"""Independent decoder-only Transformer research block.

This is an untrained, deterministic architecture primitive, not a language model.
"""
from __future__ import annotations
import math
import random


def _linear(vector, weights):
    return [sum(vector[i] * weights[i][j] for i in range(len(vector)))
            for j in range(len(weights[0]))]


def _norm(vector):
    mean = sum(vector) / len(vector)
    variance = sum((x - mean) ** 2 for x in vector) / len(vector)
    scale = 1.0 / math.sqrt(variance + 1e-5)
    return [(x - mean) * scale for x in vector]


class DecoderTransformerBlock:
    """Pre-norm decoder block: multi-head causal attention + GELU FFN."""

    def __init__(self, *, width, heads, feed_forward_width=None, seed=7):
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer.")
        if type(heads) is not int or heads <= 0 or width % heads:
            raise ValueError("heads must be positive and divide width.")
        if type(seed) is not int:
            raise TypeError("seed must be an integer.")
        hidden = feed_forward_width if feed_forward_width is not None else width * 4
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("feed_forward_width must be positive.")
        self.width, self.heads = width, heads
        self.head_width, self.feed_forward_width = width // heads, hidden
        rng = random.Random(seed)

        def init(rows, cols):
            limit = math.sqrt(6.0 / (rows + cols))
            return [[rng.uniform(-limit, limit) for _ in range(cols)]
                    for _ in range(rows)]

        self.query, self.key, self.value = (init(width, width) for _ in range(3))
        self.attention_output = init(width, width)
        self.ffn_in, self.ffn_out = init(width, hidden), init(hidden, width)

    def _attention(self, sequence):
        qs = [_linear(token, self.query) for token in sequence]
        ks = [_linear(token, self.key) for token in sequence]
        vs = [_linear(token, self.value) for token in sequence]
        outputs = []
        for position, query in enumerate(qs):
            merged = []
            for head in range(self.heads):
                start = head * self.head_width
                stop = start + self.head_width
                q = query[start:stop]
                scores = [sum(q[d] * ks[p][start + d]
                              for d in range(self.head_width)) /
                          math.sqrt(self.head_width)
                          for p in range(position + 1)]
                peak = max(scores)
                exps = [math.exp(score - peak) for score in scores]
                total = sum(exps)
                probs = [value / total for value in exps]
                for dimension in range(start, stop):
                    merged.append(sum(probs[p] * vs[p][dimension]
                                      for p in range(position + 1)))
            outputs.append(_linear(merged, self.attention_output))
        return outputs

    def forward(self, sequence):
        """Run the block. Future tokens cannot affect earlier output positions."""
        if not isinstance(sequence, list) or not sequence:
            raise ValueError("sequence must be a non-empty list.")
        if any(not isinstance(token, list) or len(token) != self.width
               for token in sequence):
            raise ValueError("sequence must have the configured width.")
        if any(type(x) not in (int, float) or not math.isfinite(x)
               for token in sequence for x in token):
            raise ValueError("sequence values must be finite numbers.")
        normalized = [_norm(token) for token in sequence]
        attended = self._attention(normalized)
        residual = [[sequence[i][j] + attended[i][j] for j in range(self.width)]
                    for i in range(len(sequence))]
        output = []
        for token in residual:
            hidden = _linear(_norm(token), self.ffn_in)
            activated = [0.5 * x * (1.0 + math.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
                         for x in hidden]
            projected = _linear(activated, self.ffn_out)
            output.append([token[i] + projected[i] for i in range(self.width)])
        return output
