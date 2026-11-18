"""Provenance-preserving next-token training and evaluation pipeline."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from .dataset import TextExample
from .tokenizer import ByteTokenizer


def audit_provenance(examples, *, approved_licenses):
    """Require an explicit human-approved license allowlist and hash source records."""
    if not isinstance(examples, list) or not examples:
        raise ValueError("At least one provenance-tagged example is required.")
    if (not isinstance(approved_licenses, (list, tuple, set))
            or not approved_licenses):
        raise ValueError("An explicit approved_licenses allowlist is required.")
    approved = {
        license_name.strip()
        for license_name in approved_licenses
        if isinstance(license_name, str) and license_name.strip()
    }
    if not approved:
        raise ValueError("approved_licenses must contain non-empty license identifiers.")

    license_counts = Counter()
    source_counts = Counter()
    digest = hashlib.sha256()
    for index, example in enumerate(examples, 1):
        if not isinstance(example, TextExample):
            raise TypeError(f"Dataset item {index} must be a TextExample.")
        if not all(isinstance(value, str) and value.strip()
                   for value in (example.text, example.source, example.license)):
            raise ValueError(f"Dataset item {index} is missing provenance metadata.")
        if example.license.strip() not in approved:
            raise ValueError(
                f"Dataset item {index} uses unapproved license: {example.license}."
            )
        license_counts[example.license.strip()] += 1
        source_counts[example.source.strip()] += 1
        for value in (example.text, example.source, example.license):
            digest.update(value.strip().encode("utf-8"))
            digest.update(bytes([0]))

    return {
        "records": len(examples),
        "approved_licenses": sorted(approved),
        "license_counts": dict(sorted(license_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "dataset_sha256": digest.hexdigest(),
    }


def build_next_token_samples(examples, tokenizer, max_sequence_length):
    """Convert licensed/provenance-tagged text examples to next-byte samples."""
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer.")
    if type(max_sequence_length) is not int or max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be a positive integer.")
    if not isinstance(examples, list) or not examples:
        raise ValueError("At least one provenance-tagged example is required.")

    samples = []
    for example in examples:
        if not isinstance(example, TextExample):
            raise TypeError("Every dataset item must be a TextExample.")
        if not all(isinstance(value, str) and value.strip()
                   for value in (example.text, example.source, example.license)):
            raise ValueError("Every example requires text, source, and license metadata.")
        tokens = tokenizer.encode(example.text)
        for target_index in range(1, len(tokens)):
            context = tokens[max(0, target_index - max_sequence_length):target_index]
            samples.append((context, tokens[target_index]))
    if not samples:
        raise ValueError("Dataset contains no next-token training pairs.")
    return samples


def train_from_examples(model, examples, *, approved_licenses, tokenizer=None,
                        epochs=1, learning_rate=0.001, max_grad_norm=1.0,
                        weight_decay=0.01):
    """Train a byte-level decoder while retaining provenance in source records."""
    audit_provenance(examples, approved_licenses=approved_licenses)
    tokenizer = tokenizer or ByteTokenizer()
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match tokenizer vocabulary size.")
    samples = build_next_token_samples(examples, tokenizer, model.max_sequence_length)
    return model.train(
        samples,
        epochs=epochs,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
    )


def evaluate_examples(model, examples, *, approved_licenses, tokenizer=None):
    """Return held-out next-token metrics after source/license allowlist audit."""
    audit_provenance(examples, approved_licenses=approved_licenses)
    tokenizer = tokenizer or ByteTokenizer()
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match tokenizer vocabulary size.")
    samples = build_next_token_samples(examples, tokenizer, model.max_sequence_length)
    total_loss = 0.0
    correct = 0
    for context, target in samples:
        logits = model.forward(context)[-1]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        denominator = sum(exps)
        total_loss -= math.log(max(exps[target] / denominator, 1e-300))
        correct += int(max(range(len(logits)), key=lambda index: logits[index]) == target)
    average_loss = total_loss / len(samples)
    perplexity = math.exp(min(average_loss, 700.0))
    return {
        "loss": average_loss,
        "perplexity": perplexity,
        "accuracy": correct / len(samples),
        "tokens": len(samples),
    }


def scheduled_learning_rate(base_rate, epoch, total_epochs, *,
                             schedule="constant", warmup_epochs=0,
                             minimum_rate=0.0):
    """Calculate a deterministic per-epoch constant, linear or cosine rate."""
    if (type(base_rate) not in (int, float) or not math.isfinite(base_rate)
            or base_rate <= 0):
        raise ValueError("base_rate must be positive and finite.")
    if type(total_epochs) is not int or total_epochs <= 0:
        raise ValueError("total_epochs must be a positive integer.")
    if type(epoch) is not int or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be within the training run.")
    if type(warmup_epochs) is not int or not 0 <= warmup_epochs < total_epochs:
        raise ValueError("warmup_epochs must be between zero and total_epochs - 1.")
    if (type(minimum_rate) not in (int, float) or not math.isfinite(minimum_rate)
            or minimum_rate < 0 or minimum_rate > base_rate):
        raise ValueError("minimum_rate must be finite and between zero and base_rate.")
    if schedule not in ("constant", "linear", "cosine"):
        raise ValueError("schedule must be constant, linear or cosine.")
    if total_epochs == 1:
        return base_rate
    if warmup_epochs and epoch <= warmup_epochs:
        return base_rate * epoch / warmup_epochs
    if schedule == "constant":
        return base_rate
    span = total_epochs - warmup_epochs - 1
    progress = 1.0 if span <= 0 else (
        epoch - warmup_epochs - 1
    ) / span
    if schedule == "linear":
        return base_rate + (minimum_rate - base_rate) * progress
    return minimum_rate + (base_rate - minimum_rate) * (
        1.0 + math.cos(math.pi * progress)
    ) / 2.0


def fit_with_validation(model, training_examples, validation_examples, *,
                        epochs=5, batch_size=None, tokenizer=None,
                        learning_rate=0.001, max_grad_norm=1.0,
                        weight_decay=0.01, optimizer="adamw", seed=0,
                        checkpoint_path=None, schedule="constant",
                        warmup_epochs=0, minimum_learning_rate=0.0,
                        approved_licenses,
                        patience=None, min_delta=0.0):
    """Train reproducibly, select the lowest-validation-loss epoch and restore it."""
    import copy
    import random

    tokenizer = tokenizer or ByteTokenizer()
    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match tokenizer vocabulary size.")
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    if patience is not None and (type(patience) is not int or patience <= 0):
        raise ValueError("patience must be a positive integer when provided.")
    if type(min_delta) not in (int, float) or not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative.")

    training_provenance = audit_provenance(
        training_examples, approved_licenses=approved_licenses
    )
    validation_provenance = audit_provenance(
        validation_examples, approved_licenses=approved_licenses
    )
    training_samples = build_next_token_samples(
        training_examples, tokenizer, model.max_sequence_length
    )
    validation_samples = build_next_token_samples(
        validation_examples, tokenizer, model.max_sequence_length
    )
    best_model = copy.deepcopy(model)
    best_loss = math.inf
    best_epoch = 0
    history = []
    rng = random.Random(seed)
    stale_epochs = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        ordered = list(training_samples)
        rng.shuffle(ordered)
        epoch_learning_rate = scheduled_learning_rate(
            learning_rate, epoch, epochs, schedule=schedule,
            warmup_epochs=warmup_epochs, minimum_rate=minimum_learning_rate,
        )
        if epoch_learning_rate == 0.0:
            train_before = evaluate_examples(
                model, training_examples, tokenizer=tokenizer,
                approved_licenses=approved_licenses,
            )["loss"]
            train_after = train_before
        else:
            train_before, train_after = model.train(
                ordered,
                epochs=1,
                batch_size=batch_size,
                learning_rate=epoch_learning_rate,
                max_grad_norm=max_grad_norm,
                weight_decay=weight_decay,
                optimizer=optimizer,
            )
        validation_metrics = evaluate_examples(
            model, validation_examples, tokenizer=tokenizer,
            approved_licenses=approved_licenses,
        )
        record = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "training_loss_before": train_before,
            "training_loss_after": train_after,
            "validation_loss": validation_metrics["loss"],
            "validation_perplexity": validation_metrics["perplexity"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_tokens": validation_metrics["tokens"],
        }
        history.append(record)
        if validation_metrics["loss"] < best_loss - min_delta:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            stale_epochs = 0
            if checkpoint_path is not None:
                best_model.save_checkpoint(checkpoint_path)
        else:
            stale_epochs += 1
            if patience is not None and stale_epochs >= patience:
                stopped_early = True
                break

    # Restore the best validation checkpoint, not merely the final epoch.
    model.__dict__.clear()
    model.__dict__.update(copy.deepcopy(best_model.__dict__))
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "training_samples": len(training_samples),
        "validation_samples": len(validation_samples),
        "training_dataset_sha256": training_provenance["dataset_sha256"],
        "validation_dataset_sha256": validation_provenance["dataset_sha256"],
    }
