import json

from special_super_intelligency.dataset import load_jsonl
from special_super_intelligency.evaluation import evaluate_regression
from special_super_intelligency.experiment import ExperimentConfig, run_experiment
from special_super_intelligency.sequence import build_next_byte_samples
from special_super_intelligency.tokenizer import ByteTokenizer


def test_end_to_end_data_tokenizer_train_evaluate_checkpoint(tmp_path):
    source = tmp_path / "training.jsonl"
    source.write_text(
        json.dumps(
            {"text": "abcabcabcabc", "source": "test fixture", "license": "CC0"}
        ) + "\n",
        encoding="utf-8",
    )
    records = load_jsonl(str(source))
    tokenized = [ByteTokenizer().encode(record.text) for record in records]
    config = ExperimentConfig(context_length=2, hidden_size=6, epochs=20, seed=5)
    model, initial_loss, final_loss = run_experiment(tokenized, config)
    assert final_loss < initial_loss

    samples = build_next_byte_samples(tokenized, context_length=2)
    metrics = evaluate_regression(model, samples)
    assert metrics.examples == len(samples)

    checkpoint = tmp_path / "integrated-model.json"
    model.save_checkpoint(str(checkpoint))
    restored = type(model).load_checkpoint(str(checkpoint))
    assert restored.predict(samples[0][0]) == model.predict(samples[0][0])
