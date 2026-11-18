"""Source-audited benchmark scoring for reasoning, coding, factuality and robustness."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re


_CATEGORIES = {"reasoning", "coding", "factuality", "robustness"}
_MANIFEST_FIELDS = {"name", "version", "split", "source", "license"}


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().casefold())


def score_benchmark(manifest, cases, predictions, *, approved_licenses):
    """Score fixed cases and predictions; never executes candidate-generated code."""
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("Benchmark manifest must declare name, version, split, source and license.")
    if any(not isinstance(value, str) or not value.strip()
           for value in manifest.values()):
        raise ValueError("Benchmark manifest fields must be non-empty strings.")
    if (not isinstance(approved_licenses, (list, tuple, set))
            or manifest["license"] not in approved_licenses):
        raise ValueError("Benchmark license is not on the explicit approved allowlist.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark cases must be a non-empty list.")
    if not isinstance(predictions, dict):
        raise TypeError("predictions must map case IDs to model outputs.")

    case_ids = []
    grouped = defaultdict(list)
    canonical_cases = []
    for case in cases:
        if not isinstance(case, dict) or not {
            "id", "category", "prompt", "expected"
        }.issubset(case):
            raise ValueError("Each case requires id, category, prompt and expected.")
        case_id = case["id"]
        category = case["category"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in case_ids:
            raise ValueError("Case IDs must be unique non-empty strings.")
        if category not in _CATEGORIES:
            raise ValueError(f"Unsupported benchmark category: {category}.")
        if not isinstance(case["prompt"], str) or not case["prompt"].strip():
            raise ValueError("Every benchmark prompt must be non-empty.")
        if category == "factuality" and (
            not isinstance(case["expected"], dict)
            or type(case["expected"].get("supported")) is not bool
        ):
            raise ValueError("Factuality expected value must declare a boolean supported label.")
        if category == "robustness":
            if (not isinstance(case.get("pair_id"), str) or not case["pair_id"].strip()
                    or type(case.get("expect_same")) is not bool):
                raise ValueError("Robustness cases require pair_id and expect_same.")
            grouped[case["pair_id"]].append(case)
        case_ids.append(case_id)
        canonical_cases.append(case)

    if set(predictions) != set(case_ids):
        raise ValueError("Predictions must contain exactly one output for every benchmark case.")
    for pair_id, pair in grouped.items():
        if len(pair) != 2 or pair[0]["expect_same"] != pair[1]["expect_same"]:
            raise ValueError(f"Robustness pair {pair_id} must contain exactly two matching cases.")

    digest_payload = {
        "manifest": manifest,
        "cases": sorted(canonical_cases, key=lambda item: item["id"]),
    }
    dataset_hash = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    category_scores = {}
    for category in sorted(_CATEGORIES):
        selected = [case for case in cases if case["category"] == category]
        if not selected:
            continue
        if category in ("reasoning", "coding"):
            correct = sum(
                _normalize(str(predictions[case["id"]])) == _normalize(str(case["expected"]))
                for case in selected
            )
            category_scores[category] = {
                "cases": len(selected), "accuracy": correct / len(selected)
            }
        elif category == "factuality":
            correct = 0
            cited = 0
            for case in selected:
                prediction = predictions[case["id"]]
                if not isinstance(prediction, dict) or type(prediction.get("supported")) is not bool:
                    raise ValueError("Factuality predictions require a boolean supported field.")
                citations = prediction.get("citations", [])
                if not isinstance(citations, list):
                    raise TypeError("Factuality citations must be a list.")
                correct += int(prediction["supported"] == case["expected"]["supported"])
                cited += int(bool(citations))
            category_scores[category] = {
                "cases": len(selected),
                "support_accuracy": correct / len(selected),
                "citation_coverage": cited / len(selected),
            }
        else:
            pair_results = []
            for pair in grouped.values():
                first, second = pair
                same = _normalize(str(predictions[first["id"]])) == _normalize(
                    str(predictions[second["id"]])
                )
                pair_results.append(same == first["expect_same"])
            category_scores[category] = {
                "pairs": len(pair_results),
                "consistency_accuracy": sum(pair_results) / len(pair_results),
            }

    return {
        "benchmark": manifest["name"],
        "version": manifest["version"],
        "split": manifest["split"],
        "dataset_sha256": dataset_hash,
        "cases": len(cases),
        "category_scores": category_scores,
    }


def compare_benchmarks(candidate, baseline):
    """Compare reports only when they share the exact benchmark dataset hash."""
    if candidate.get("dataset_sha256") != baseline.get("dataset_sha256"):
        raise ValueError("Cannot compare benchmark reports from different datasets.")
    deltas = {}
    for category, candidate_metrics in candidate["category_scores"].items():
        if category not in baseline["category_scores"]:
            raise ValueError("Benchmark category sets do not match.")
        baseline_metrics = baseline["category_scores"][category]
        metric = next(
            (key for key in ("accuracy", "support_accuracy", "consistency_accuracy")
             if key in candidate_metrics),
            None,
        )
        if metric is not None:
            deltas[category] = candidate_metrics[metric] - baseline_metrics[metric]
    return {"dataset_sha256": candidate["dataset_sha256"], "score_deltas": deltas}


def load_benchmark_bundle(path):
    """Load a versioned JSON benchmark bundle with cases and baseline outputs."""
    from pathlib import Path

    source = Path(path)
    size = source.stat().st_size
    if size <= 0 or size > 10 * 1024 * 1024:
        raise ValueError("Benchmark bundle size is invalid.")
    bundle = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(bundle, dict) or set(bundle) != {
        "manifest", "cases", "baseline_predictions"
    }:
        raise ValueError("Benchmark bundle structure is invalid.")
    if not isinstance(bundle["baseline_predictions"], dict):
        raise TypeError("Benchmark baseline predictions must be an object.")
    return bundle
