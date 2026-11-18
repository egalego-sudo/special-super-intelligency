import math

import pytest

from special_super_intelligency.benchmark import benchmark_decoder_inference
from special_super_intelligency.transformer import DecoderLanguageModel


def test_benchmark_compares_equivalent_cached_and_recompute_paths():
    model = DecoderLanguageModel(
        vocab_size=12, width=4, heads=2, layers=1, max_sequence_length=8, seed=179
    )
    report = benchmark_decoder_inference(model, [1, 2, 3], new_tokens=3, repeats=1)
    assert report["outputs_match"] is True
    assert report["generated_tokens"] == 3
    assert report["model"]["parameters"] > 0
    assert report["hardware"]["python"]
    assert report["cached_peak_python_bytes"] > 0
    assert report["recompute_peak_python_bytes"] > 0
    assert math.isfinite(report["cached_tokens_per_second"])
    assert math.isfinite(report["recompute_tokens_per_second"])


@pytest.mark.parametrize("kwargs", [
    {"new_tokens": 0},
    {"repeats": 0},
])
def test_benchmark_rejects_invalid_settings(kwargs):
    model = DecoderLanguageModel(vocab_size=8, width=4, heads=2, layers=1)
    with pytest.raises(ValueError):
        benchmark_decoder_inference(model, [1, 2], **kwargs)
