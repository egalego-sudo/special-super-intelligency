"""Provenance-preserving next-token training and evaluation pipeline."""
from __future__ import annotations

import math

from .dataset import TextExample
from .tokenizer import ByteTokenizer


def build_next_token_samples(examples, tokenizer, max_sequence_length):
    """Convert licensed/provenance-tagged text examples to next-byte samples."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer.")
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be a positive integer.")
    if not isinstance(examples, list) or not examples:
        raise ValueError("At least one provenance-tagged example is required.")

    samples = []
    for example in examples:
        if not isinstance(example, TextExample):
            raise TypeError("Every dataset item must be a TextExample.")
        if not all(isinstance(value, str) and value.strip()
                   for value in (example.text, example.source, example.license)):
            raise ValueError("Every example requires text, source, and license metadata.")
        tokens = tokenizer.encode(example.text)
        for target_index in range(1, len(tokens)):
            context = tokens[max(0, target_index - max_sequence_length):target_index]
            samples.append((context, tokens[target_index]))
    if not samples:
        raise ValueError("Dataset contains no next-token training pairs.")
    return samples


def train_from_examples(model, examples, *, tokenizer=None, epochs=1,
                        learning_rate=0.001, max_grad_norm=1.0,
                        weight_decay=0.01):
    """Train a byte-level decoder while retaining provenance in source records."""
    tokenizer = tokenizer or ByteTokenizer()
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match tokenizer vocabulary size.")
    samples = build_next_token_samples(examples, tokenizer, model.max_sequence_length)
    return model.train(
        samples,
        epochs=epochs,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
    )


def evaluate_examples(model, examples, *, tokenizer=None):
    """Return held-out next-token loss, perplexity, accuracy and token count."""
    tokenizer = tokenizer or ByteTokenizer()
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match tokenizer vocabulary size.")
    samples = build_next_token_samples(examples, tokenizer, model.max_sequence_length)
    total_loss = 0.0
    correct = 0
    for context, target in samples:
        logits = model.forward(context)[-1]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        denominator = sum(exps)
        total_loss -= math.log(max(exps[target] / denominator, 1e-300))
        correct += int(max(range(len(logits)), key=lambda index: logits[index]) == target)
    average_loss = total_loss / len(samples)
    perplexity = math.exp(min(average_loss, 700.0))
    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "accuracy": correct / len(samples),
        "tokens": len(samples),
    }
