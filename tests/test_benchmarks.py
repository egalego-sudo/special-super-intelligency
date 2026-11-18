import copy

import pytest

from special_super_intelligency.benchmarks import compare_benchmarks, score_benchmark


MANIFEST = {
    "name": "QUANTUM AI synthetic capability smoke",
    "version": "1.0",
    "split": "test",
    "source": "author-created unit fixtures",
    "license": "CC0",
}
CASES = [
    {"id": "r1", "category": "reasoning", "prompt": "What is 2+2?", "expected": "4"},
    {"id": "c1", "category": "coding", "prompt": "Return output of a tiny function", "expected": "hello"},
    {"id": "f1", "category": "factuality", "prompt": "Is the claim supported?", "expected": {"supported": True}},
    {"id": "rob1a", "category": "robustness", "prompt": "Paraphrase A", "expected": "", "pair_id": "same-1", "expect_same": True},
    {"id": "rob1b", "category": "robustness", "prompt": "Paraphrase B", "expected": "", "pair_id": "same-1", "expect_same": True},
    {"id": "rob2a", "category": "robustness", "prompt": "Constraint A", "expected": "", "pair_id": "different-1", "expect_same": False},
    {"id": "rob2b", "category": "robustness", "prompt": "Constraint B", "expected": "", "pair_id": "different-1", "expect_same": False},
]
PREDICTIONS = {
    "r1": "4",
    "c1": "hello",
    "f1": {"supported": True, "citations": ["fixture-source-1"]},
    "rob1a": "same answer",
    "rob1b": "same answer",
    "rob2a": "short response",
    "rob2b": "long response",
}


def test_benchmark_scores_reasoning_coding_factuality_and_robustness():
    report = score_benchmark(
        MANIFEST, CASES, PREDICTIONS, approved_licenses=("CC0",)
    )
    assert report["cases"] == 7
    assert report["category_scores"]["reasoning"]["accuracy"] == 1.0
    assert report["category_scores"]["coding"]["accuracy"] == 1.0
    assert report["category_scores"]["factuality"]["support_accuracy"] == 1.0
    assert report["category_scores"]["factuality"]["citation_coverage"] == 1.0
    assert report["category_scores"]["robustness"]["consistency_accuracy"] == 1.0
    assert len(report["dataset_sha256"]) == 64


def test_benchmark_baseline_comparison_requires_identical_dataset():
    baseline_predictions = dict(PREDICTIONS)
    baseline_predictions["r1"] = "wrong"
    baseline = score_benchmark(
        MANIFEST, CASES, baseline_predictions, approved_licenses=("CC0",)
    )
    candidate = score_benchmark(
        MANIFEST, CASES, PREDICTIONS, approved_licenses=("CC0",)
    )
    comparison = compare_benchmarks(candidate, baseline)
    assert comparison["score_deltas"]["reasoning"] == 0.5


def test_benchmark_rejects_unapproved_license_and_missing_predictions():
    with pytest.raises(ValueError, match="allowlist"):
        score_benchmark(MANIFEST, CASES, PREDICTIONS, approved_licenses=("MIT",))
    incomplete = dict(PREDICTIONS)
    incomplete.pop("r1")
    with pytest.raises(ValueError, match="exactly one"):
        score_benchmark(MANIFEST, CASES, incomplete, approved_licenses=("CC0",))


def test_benchmark_rejects_incomplete_robustness_pairs():
    cases = [case for case in CASES if case["id"] != "rob1b"]
    predictions = {key: value for key, value in PREDICTIONS.items() if key != "rob1b"}
    with pytest.raises(ValueError, match="exactly two"):
        score_benchmark(MANIFEST, cases, predictions, approved_licenses=("CC0",))


def test_benchmark_comparison_rejects_different_datasets():
    first = score_benchmark(MANIFEST, CASES, PREDICTIONS, approved_licenses=("CC0",))
    changed_manifest = copy.deepcopy(MANIFEST)
    changed_manifest["version"] = "2.0"
    second = score_benchmark(
        changed_manifest, CASES, PREDICTIONS, approved_licenses=("CC0",)
    )
    with pytest.raises(ValueError, match="different datasets"):
        compare_benchmarks(first, second)
