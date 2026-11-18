"""Deterministic UTF-8 byte tokenizer baseline.

This is a transparent tokenizer baseline, not a learned BPE or production tokenizer.
IDs 0..255 map directly to byte values, making encoding reversible for all Unicode text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """Encode text as UTF-8 bytes and decode token IDs back to text."""

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list of integers")
        if any(type(token_id) is not int or not 0 <= token_id < self.vocab_size
               for token_id in token_ids):
            raise ValueError("token IDs must be integers in the range 0..255")
        try:
            return bytes(token_ids).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("token IDs do not form valid UTF-8 text") from exc
