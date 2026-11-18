"""Versioned, atomic JSON checkpoints for the decoder research model."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

FORMAT_VERSION = 1
MAX_CHECKPOINT_BYTES = 100 * 1024 * 1024
_BLOCK_PARAMETERS = (
    "query", "key", "value", "attention_output", "ffn_in", "ffn_out"
)


def _matrix(matrix, rows, columns, name):
    if not isinstance(matrix, list) or len(matrix) != rows:
        raise ValueError(f"{name} has an invalid row count.")
    checked = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{name} has an invalid column count.")
        values = []
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} contains a non-finite or non-numeric value.")
            values.append(float(value))
        checked.append(values)
    return checked


def save_checkpoint(model, path):
    """Atomically persist model configuration and every learned parameter."""
    target = Path(path)
    if not target.parent.exists() or not target.parent.is_dir():
        raise ValueError("Checkpoint parent directory must already exist.")
    blocks = []
    for block in model.blocks:
        blocks.append({
            name: getattr(block, name)
            for name in _BLOCK_PARAMETERS
        })
    payload = {
        "format_version": FORMAT_VERSION,
        "model": {
            "vocab_size": model.vocab_size,
            "width": model.width,
            "heads": model.blocks[0].heads,
            "layers": len(model.blocks),
            "max_sequence_length": model.max_sequence_length,
        },
        "state": {
            "embedding": model.embedding,
            "output_projection": model.output_projection,
            "blocks": blocks,
        },
    }
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint exceeds the configured size limit.")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    """Load a validated checkpoint into a fresh DecoderLanguageModel."""
    from .transformer import DecoderLanguageModel

    source = Path(path)
    size = source.stat().st_size
    if size <= 0 or size > MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint file size is invalid.")
    raw = source.read_text(encoding="utf-8")

    def reject_constant(value):
        raise ValueError(f"Invalid non-finite JSON constant: {value}")

    payload = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(payload, dict) or set(payload) != {
        "format_version", "model", "state"
    }:
        raise ValueError("Checkpoint structure is invalid.")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version.")
    config = payload["model"]
    if not isinstance(config, dict) or set(config) != {
        "vocab_size", "width", "heads", "layers", "max_sequence_length"
    }:
        raise ValueError("Checkpoint model configuration is invalid.")

    model = DecoderLanguageModel(
        vocab_size=config["vocab_size"],
        width=config["width"],
        heads=config["heads"],
        layers=config["layers"],
        max_sequence_length=config["max_sequence_length"],
        seed=0,
    )
    state = payload["state"]
    if not isinstance(state, dict) or set(state) != {
        "embedding", "output_projection", "blocks"
    }:
        raise ValueError("Checkpoint state structure is invalid.")
    model.embedding = _matrix(
        state["embedding"], model.vocab_size, model.width, "embedding"
    )
    model.output_projection = _matrix(
        state["output_projection"], model.width, model.vocab_size,
        "output_projection"
    )
    if not isinstance(state["blocks"], list) or len(state["blocks"]) != len(model.blocks):
        raise ValueError("Checkpoint decoder block count is invalid.")

    for index, (saved, block) in enumerate(zip(state["blocks"], model.blocks)):
        if not isinstance(saved, dict) or set(saved) != set(_BLOCK_PARAMETERS):
            raise ValueError(f"Checkpoint block {index} structure is invalid.")
        shapes = {
            "query": (model.width, model.width),
            "key": (model.width, model.width),
            "value": (model.width, model.width),
            "attention_output": (model.width, model.width),
            "ffn_in": (model.width, block.feed_forward_width),
            "ffn_out": (block.feed_forward_width, model.width),
        }
        for name, (rows, columns) in shapes.items():
            setattr(block, name, _matrix(
                saved[name], rows, columns, f"blocks[{index}].{name}"
            ))
    return model
