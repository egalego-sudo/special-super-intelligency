import pytest

from special_super_intelligency.experiment import ExperimentConfig, run_experiment


def test_experiment_config_is_reproducible():
    config = ExperimentConfig(context_length=2, hidden_size=4, epochs=5, seed=17)
    first = run_experiment([[1, 2, 3, 1, 2, 3]], config)
    second = run_experiment([[1, 2, 3, 1, 2, 3]], config)
    assert first[1:] == second[1:]
    assert first[0].predict([1 / 255, 2 / 255]) == second[0].predict([1 / 255, 2 / 255])


def test_experiment_rejects_invalid_config():
    with pytest.raises(ValueError):
        run_experiment([[1, 2]], ExperimentConfig(epochs=0))
