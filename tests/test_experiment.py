import pytest

from special_super_intelligency.experiment import (
    ExperimentConfig,
    run_decoder_seed_sweep,
    run_experiment,
)


def test_experiment_config_is_reproducible():
    config = ExperimentConfig(context_length=2, hidden_size=4, epochs=5, seed=17)
    first = run_experiment([[1, 2, 3, 1, 2, 3]], config)
    second = run_experiment([[1, 2, 3, 1, 2, 3]], config)
    assert first[1:] == second[1:]
    assert first[0].predict([1 / 255, 2 / 255]) == second[0].predict([1 / 255, 2 / 255])


def test_experiment_rejects_invalid_config():
    with pytest.raises(ValueError):
        run_experiment([[1, 2]], ExperimentConfig(epochs=0))


def test_multi_seed_sweep_records_variance_provenance_and_atomic_report(tmp_path):
    destination = tmp_path / "artifacts" / "seed-sweep.json"
    destination.parent.mkdir()
    report = run_decoder_seed_sweep(
        seeds=(131, 137), epochs=3, output_path=destination
    )
    assert report["synthetic_only"] is True
    assert report["seed_count"] == 2
    assert report["seeds"] == [131, 137]
    assert len(report["runs"]) == 2
    assert len(report["report_sha256"]) == 64
    assert len(report["training_provenance"]["dataset_sha256"]) == 64
    assert len(report["validation_provenance"]["dataset_sha256"]) == 64
    assert report["aggregate"]["validation_loss_after"]["sample_stddev"] >= 0
    import json

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved == report


def test_multi_seed_sweep_is_reproducible_for_same_seed_set():
    first = run_decoder_seed_sweep(seeds=(149, 151), epochs=2)
    second = run_decoder_seed_sweep(seeds=(149, 151), epochs=2)
    assert first == second


@pytest.mark.parametrize("seeds", [(1,), (1, 1), (-1, 2), (True, 2), "1,2"])
def test_multi_seed_sweep_rejects_invalid_seed_sets(seeds):
    with pytest.raises(ValueError, match="unique non-negative"):
        run_decoder_seed_sweep(seeds=seeds, epochs=1)
