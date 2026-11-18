"""Independent decoder-only Transformer research block.

This is an untrained, deterministic architecture primitive, not a language model.
"""
from __future__ import annotations

import copy
import math
import random

from .autodiff import Value
from .optim import AdamW


def _linear(vector, weights):
    return [sum(vector[i] * weights[i][j] for i in range(len(vector)))
            for j in range(len(weights[0]))]


def _exp(value):
    return value.exp() if isinstance(value, Value) else math.exp(value)


def _tanh(value):
    return value.tanh() if isinstance(value, Value) else math.tanh(value)


def _norm(vector):
    mean = sum(vector) / len(vector)
    variance = sum((x - mean) ** 2 for x in vector) / len(vector)
    scale = (variance + 1e-5) ** -0.5
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
                exps = [_exp(score - peak) for score in scores]
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
        if any(type(x.data if isinstance(x, Value) else x) not in (int, float)
               or not math.isfinite(x.data if isinstance(x, Value) else x)
               for token in sequence for x in token):
            raise ValueError("sequence values must be finite numbers.")
        normalized = [_norm(token) for token in sequence]
        attended = self._attention(normalized)
        residual = [[sequence[i][j] + attended[i][j] for j in range(self.width)]
                    for i in range(len(sequence))]
        output = []
        for token in residual:
            hidden = _linear(_norm(token), self.ffn_in)
            activated = [
                0.5 * x * (1.0 + _tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
                ))
                for x in hidden
            ]
            projected = _linear(activated, self.ffn_out)
            output.append([token[i] + projected[i] for i in range(self.width)])
        return output


class DecoderLanguageModel:
    """Small untrained decoder-only language-model architecture.

    This composes token embeddings, sinusoidal positions, decoder blocks, and a
    vocabulary projection. Parameters are initialized randomly; no pretrained
    weights or training capability are implied by construction alone.
    """

    def __init__(self, *, vocab_size, width, heads, layers=2,
                 max_sequence_length=128, seed=7):
        for name, value in (("vocab_size", vocab_size), ("width", width),
                            ("heads", heads), ("layers", layers),
                            ("max_sequence_length", max_sequence_length)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if width % heads:
            raise ValueError("heads must divide width exactly.")
        if type(seed) is not int:
            raise TypeError("seed must be an integer.")
        self.vocab_size = vocab_size
        self.width = width
        self.max_sequence_length = max_sequence_length
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(width)
        self.embedding = [
            [rng.uniform(-scale, scale) for _ in range(width)]
            for _ in range(vocab_size)
        ]
        self.blocks = [
            DecoderTransformerBlock(width=width, heads=heads, seed=seed + i + 1)
            for i in range(layers)
        ]
        self.output_projection = [
            [rng.uniform(-scale, scale) for _ in range(vocab_size)]
            for _ in range(width)
        ]

    def _encode(self, token_ids):
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("token_ids must be a non-empty list.")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("token sequence exceeds max_sequence_length.")
        if any(type(token) is not int or not 0 <= token < self.vocab_size
               for token in token_ids):
            raise ValueError("token IDs must be integers within the vocabulary.")
        sequence = []
        for position, token_id in enumerate(token_ids):
            vector = []
            for dimension in range(self.width):
                angle = position / (10000 ** (2 * (dimension // 2) / self.width))
                positional = math.sin(angle) if dimension % 2 == 0 else math.cos(angle)
                vector.append(self.embedding[token_id][dimension] + positional)
            sequence.append(vector)
        for block in self.blocks:
            sequence = block.forward(sequence)
        return sequence

    def forward(self, token_ids):
        """Return per-position vocabulary logits for a non-empty token sequence."""
        return [_linear(token, self.output_projection)
                for token in self._encode(token_ids)]

    def next_token_probabilities(self, token_ids):
        """Return a stable softmax distribution for the final position."""
        logits = self.forward(token_ids)[-1]
        peak = max(logits)
        exponentials = [_exp(value - peak) for value in logits]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def generate(self, prompt, *, max_new_tokens=1):
        """Greedy autoregressive decoding; outputs are untrained until trained."""
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("prompt must be a non-empty list.")
        generated = list(prompt)
        for _ in range(max_new_tokens):
            context = generated[-self.max_sequence_length:]
            probabilities = self.next_token_probabilities(context)
            generated.append(max(range(self.vocab_size),
                                 key=lambda index: probabilities[index]))
        return generated

    def train_output_head(self, samples, *, epochs=100, learning_rate=0.1):
        """Train vocabulary projection with next-token cross-entropy.

        Decoder blocks and token embeddings remain frozen in this first training
        milestone. Samples are (non-empty context token IDs, target token ID).
        """
        if not isinstance(samples, list) or not samples:
            raise ValueError("At least one training sample is required.")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if (type(learning_rate) not in (int, float)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite.")
        validated = []
        for context, target in samples:
            self._encode(context)
            if type(target) is not int or not 0 <= target < self.vocab_size:
                raise ValueError("target token must be within the vocabulary.")
            validated.append((context, target))

        def calculate_loss():
            total = 0.0
            for context, target in validated:
                hidden = self._encode(context)[-1]
                logits = _linear(hidden, self.output_projection)
                peak = max(logits)
                exps = [math.exp(value - peak) for value in logits]
                total += -math.log(max(exps[target] / sum(exps), 1e-300))
            return total / len(validated)

        initial_loss = calculate_loss()
        for _ in range(epochs):
            gradients = [[0.0] * self.vocab_size for _ in range(self.width)]
            count = len(validated)
            for context, target in validated:
                hidden = self._encode(context)[-1]
                logits = _linear(hidden, self.output_projection)
                peak = max(logits)
                exps = [math.exp(value - peak) for value in logits]
                total = sum(exps)
                delta = [value / total for value in exps]
                delta[target] -= 1.0
                for row in range(self.width):
                    for column in range(self.vocab_size):
                        gradients[row][column] += hidden[row] * delta[column] / count
            for row in range(self.width):
                for column in range(self.vocab_size):
                    self.output_projection[row][column] -= (
                        learning_rate * gradients[row][column]
                    )
        return initial_loss, calculate_loss()


    def train(self, samples, *, epochs=10, learning_rate=0.001,
              max_grad_norm=1.0, weight_decay=0.01):
        """Train every decoder parameter with reverse-mode autodiff and AdamW.

        Samples are (context token IDs, target token ID). This deliberately
        small reference trainer favors correctness and inspectability over speed.
        """
        if not isinstance(samples, list) or not samples:
            raise ValueError("At least one training sample is required.")
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        validated = []
        for context, target in samples:
            self._encode(context)
            if type(target) is not int or not 0 <= target < self.vocab_size:
                raise ValueError("target token must be within the vocabulary.")
            validated.append((list(context), target))

        def measure_loss():
            total_loss = 0.0
            for context, target in validated:
                logits = self.forward(context)[-1]
                peak = max(logits)
                exps = [math.exp(value - peak) for value in logits]
                total_loss -= math.log(max(exps[target] / sum(exps), 1e-300))
            return total_loss / len(validated)

        def parameter_groups(model):
            groups = [model.embedding, model.output_projection]
            for block in model.blocks:
                for name in ("query", "key", "value", "attention_output",
                             "ffn_in", "ffn_out"):
                    groups.append(getattr(block, name))
            return groups

        tracked = copy.deepcopy(self)
        groups = parameter_groups(tracked)
        parameters = []
        for matrix in groups:
            for row_index, row in enumerate(matrix):
                matrix[row_index] = [Value(value) for value in row]
                parameters.extend(matrix[row_index])

        optimizer = AdamW(
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        )
        initial_loss = measure_loss()
        for _ in range(epochs):
            for parameter in parameters:
                parameter.grad = 0.0
            loss = Value(0.0)
            for context, target in validated:
                logits = tracked.forward(context)[-1]
                peak = max(logits, key=lambda item: item.data)
                exps = [_exp(value - peak) for value in logits]
                total = sum(exps)
                loss = loss - (exps[target] / total).log() / len(validated)
            loss.backward()
            optimizer.step(parameters)

        # Copy the learned scalar values back to the ordinary inference model.
        destination_groups = parameter_groups(self)
        for destination, learned in zip(destination_groups, groups):
            for row_index, row in enumerate(learned):
                destination[row_index] = [parameter.data for parameter in row]

        return initial_loss, measure_loss()
