"""Auditable JSONL text dataset loader with mandatory provenance metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import ByteTokenizer


@dataclass(frozen=True)
class TextExample:
    text: str
    source: str
    license: str


def load_jsonl(path: str) -> list[TextExample]:
    """Load examples only when text, source, and license are explicitly present."""
    examples: list[TextExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}.") from exc
        if not isinstance(record, dict):
            raise TypeError(f"Line {line_number} must contain a JSON object.")
        text = record.get("text")
        source = record.get("source")
        license_name = record.get("license")
        if not all(isinstance(value, str) and value.strip()
                   for value in (text, source, license_name)):
            raise ValueError(
                f"Line {line_number} requires non-empty text, source, and license fields."
            )
        examples.append(TextExample(text.strip(), source.strip(), license_name.strip()))
    return examples


def encode_examples(examples: list[TextExample], tokenizer: ByteTokenizer) -> list[list[int]]:
    """Encode validated examples without discarding their source records."""
    return [tokenizer.encode(example.text) for example in examples]
