"""Reproducible inference latency and Python-memory benchmark harness."""
from __future__ import annotations

import os
import platform
import statistics
import sys
import time
import tracemalloc


def _parameter_count(model):
    matrices = [model.embedding, model.output_projection]
    for block in model.blocks:
        matrices.extend(
            getattr(block, name)
            for name in ("query", "key", "value", "attention_output", "ffn_in", "ffn_out")
        )
    return sum(len(row) for matrix in matrices for row in matrix)


def _recompute_greedy(model, prompt, new_tokens):
    generated = list(prompt)
    for _ in range(new_tokens):
        logits = model.forward(generated[-model.max_sequence_length:])[-1]
        generated.append(max(range(model.vocab_size), key=lambda index: logits[index]))
    return generated


def _measure(function, repeats):
    durations = []
    peak_memory = 0
    result = None
    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory = max(peak_memory, peak)
    return result, statistics.median(durations), peak_memory


def benchmark_decoder_inference(model, prompt, *, new_tokens=16, repeats=3):
    """Compare KV-cached generation with full-prefix recomputation.

    The report is descriptive and hardware-labelled. It intentionally makes no
    pass/fail speed claim because timings vary with runner load and hardware.
    """
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty list.")
    if type(new_tokens) is not int or new_tokens <= 0:
        raise ValueError("new_tokens must be a positive integer.")
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be a positive integer.")
    model.forward(prompt[-model.max_sequence_length:])

    # Warm both paths once outside the timed samples.
    cached_expected = model.generate(prompt, max_new_tokens=new_tokens)
    recompute_expected = _recompute_greedy(model, prompt, new_tokens)
    if cached_expected != recompute_expected:
        raise AssertionError("KV-cached and recomputed generation differ.")

    cached, cached_seconds, cached_peak = _measure(
        lambda: model.generate(prompt, max_new_tokens=new_tokens), repeats
    )
    recomputed, recompute_seconds, recompute_peak = _measure(
        lambda: _recompute_greedy(model, prompt, new_tokens), repeats
    )
    if cached != recomputed:
        raise AssertionError("Benchmark paths produced different token sequences.")

    return {
        "benchmark": "decoder-kv-cache-vs-full-recompute-v1",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported",
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "model": {
            "vocab_size": model.vocab_size,
            "width": model.width,
            "layers": len(model.blocks),
            "heads": model.blocks[0].heads,
            "parameters": _parameter_count(model),
        },
        "prompt_tokens": len(prompt),
        "generated_tokens": new_tokens,
        "repeats": repeats,
        "cached_median_seconds": cached_seconds,
        "recompute_median_seconds": recompute_seconds,
        "cached_tokens_per_second": new_tokens / cached_seconds,
        "recompute_tokens_per_second": new_tokens / recompute_seconds,
        "cached_peak_python_bytes": cached_peak,
        "recompute_peak_python_bytes": recompute_peak,
        "outputs_match": cached == recomputed,
    }
