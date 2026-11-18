"""Right-padding utilities and attention masks for variable-length batches."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaddedBatch:
    input_ids: list[list[int]]
    attention_mask: list[list[bool]]
    lengths: list[int]


def pad_sequences(sequences, *, pad_token_id=0, max_length=None):
    """Right-pad non-empty token sequences and return explicit boolean masks."""
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("sequences must be a non-empty list.")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer.")
    if any(not isinstance(sequence, list) or not sequence
           or any(type(token) is not int or token < 0 for token in sequence)
           for sequence in sequences):
        raise ValueError("Every sequence must contain non-negative integer token IDs.")
    longest = max(len(sequence) for sequence in sequences)
    target_length = longest if max_length is None else max_length
    if type(target_length) is not int or target_length < longest or target_length <= 0:
        raise ValueError("max_length must be an integer at least as long as every sequence.")
    padded, masks, lengths = [], [], []
    for sequence in sequences:
        length = len(sequence)
        padding = target_length - length
        padded.append(sequence + [pad_token_id] * padding)
        masks.append([True] * length + [False] * padding)
        lengths.append(length)
    return PaddedBatch(padded, masks, lengths)


def pad_next_token_samples(samples, *, pad_token_id=0, max_length=None):
    """Pad (context, target) pairs into (context, mask, target) training rows."""
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list.")
    contexts, targets = [], []
    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("Each sample must contain a context and target.")
        context, target = sample
        if type(target) is not int or target < 0:
            raise ValueError("Targets must be non-negative integer token IDs.")
        contexts.append(context)
        targets.append(target)
    batch = pad_sequences(contexts, pad_token_id=pad_token_id, max_length=max_length)
    return [
        (context, mask, target)
        for context, mask, target in zip(batch.input_ids, batch.attention_mask, targets)
    ]
