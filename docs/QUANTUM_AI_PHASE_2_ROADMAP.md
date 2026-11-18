# QUANTUM AI — Independent Foundation Model Phase 2 Roadmap

**Scope:** evolve the repository's educational neural-network baseline into a genuinely trainable, independently parameterized decoder language model.
**Tracking:** 20 equal acceptance packages; one verified package = 5%. This percentage measures this Phase 2 engineering plan only, never intelligence or capability versus GPT-6.
**Current branch:** `android-control-app-v1`

| ID | Acceptance package | Status | Evidence / exit condition |
|---|---|---|---|
| 01 | Implement multi-head causal decoder block with pre-norm, residual and feed-forward paths | COMPLETE | `transformer.py`; causal-mask, shape, determinism and input-validation tests pass in CI |
| 02 | Compose token embeddings, positional encoding, stacked decoder blocks and vocabulary projection | COMPLETE | `DecoderLanguageModel`; logits, softmax, generation, context-limit and future-token isolation tests pass in CI |
| 03 | Add supervised next-token training for the vocabulary output head | COMPLETE | Cross-entropy SGD training reduces loss on a deterministic tiny dataset; invalid target test; CI pass |
| 04 | Implement end-to-end backpropagation through embeddings, attention and feed-forward blocks | COMPLETE | `autodiff.py` reverse-mode engine; model trainer updates embeddings, Q/K/V attention, attention output, FFN and vocabulary projection; scalar and full-model finite-difference checks; CI 98 tests passed |
| 05 | Add AdamW optimizer, gradient clipping and numerical stability checks | COMPLETE | `optim.py` AdamW with bias correction, decoupled weight decay, global norm clipping and non-finite guards; integrated into model trainer; optimizer and end-to-end training CI tests pass |
| 06 | Add full-model versioned checkpoint save/load | COMPLETE | Atomic versioned JSON checkpoint; full-parameter round-trip output equivalence; malformed JSON, unsupported version, non-finite weight and wrong-shape rejection; CI pass |
| 07 | Integrate byte tokenizer and provenance-checked text dataset into the decoder model | COMPLETE | `training.py` converts provenance-tagged TextExample records into byte-level next-token samples; model `train_text` and `evaluate_text` APIs; integration, metadata rejection, vocabulary mismatch and metric tests; CI 108 passed |
| 08 | Add minibatching, padding and causal/padding attention masks | COMPLETE | `batching.py` right-padding masks; decoder attention excludes padded keys; masked `forward_batch`; variable-size minibatch training; padding leakage and batch-equivalence tests; CI 115 passed |
| 09 | Add reproducible train/validation training loop with checkpoint selection | COMPLETE | `fit` separates tagged training/validation samples, deterministic seed shuffle, epoch metrics, best validation-loss model selection, checkpoint save and in-memory restore; repeatability and checkpoint-equivalence tests; CI 117 passed |
| 10 | Add learning-rate schedule and early stopping | COMPLETE | Constant/linear/cosine schedules, optional warmup, min-delta and patience-based early stopping; best-checkpoint restore; boundary and stopping tests; CI 123 passed |
| 11 | Demonstrate controlled tiny-corpus overfit and held-out generalization measurement | COMPLETE | `run_decoder_tiny_corpus_experiment` uses separate authored synthetic train/validation fixtures, fixed seed and reproducible training settings; reports train loss change, held-out loss, perplexity, accuracy and generalization gap; CI test confirms overfit and finite held-out metrics |
| 12 | Add language-model perplexity and next-token accuracy evaluation | COMPLETE | Held-out evaluator returns cross-entropy loss, perplexity, next-token accuracy and token count; metric bounds and dataset integration tests pass |
| 13 | Add licensed corpus ingestion and source-level provenance audit | COMPLETE | Training, evaluation and fit require an explicit human-approved license allowlist; exact license/source counts and SHA-256 dataset digests recorded separately for train/validation; unapproved license rejection tests; human legal review remains mandatory |
| 14 | Add configurable temperature, top-k and top-p sampling | COMPLETE | Temperature scaling, top-k truncation, nucleus top-p filtering, seeded sampling and autoregressive generation; stable normalized probabilities, greedy-default compatibility and invalid-parameter tests; CI 128 passed |
| 15 | Add efficient inference context handling and KV cache | COMPLETE | Per-layer incremental key/value cache, one-token forward path, rolling context rebuild at sequence limit; cached logits match full recomputation; overflow and malformed-cache tests; CI 142 passed |
| 16 | Add performance profiling and memory/throughput baselines | COMPLETE | `benchmark.py` reports hardware/Python, model parameter count, cached vs full-recompute median latency, tokens/sec and peak Python allocation; asserts output equivalence without imposing noisy speed thresholds; CI 145 passed |
| 17 | Add broad reasoning, coding, factuality and robustness benchmark harness | COMPLETE | `benchmarks.py` scores reasoning/coding exact-match, factuality support labels plus evidence-ID citation precision/coverage, and paired robustness; immutable dataset hashes and strict category/metric/sample-count baseline comparison; licensed synthetic smoke bundle; regression CI 164 passed |
| 18 | Add adversarial, prompt-injection and unsafe-tool-use evaluation | COMPLETE | Threat model documented; prompt-injection approval bypass tests, sandbox image/container flag injection rejection, no-network/read-only/non-root isolation, bounded resources and fail-closed tests; CI 164 passed and Ruff passed. Formal penetration testing remains NOT VERIFIED |
| 19 | Run multiple-seed experiments and preserve full experiment provenance | COMPLETE | `run_decoder_seed_sweep` runs repeatable seeds, reports mean/sample standard deviation/min/max, records separate train/validation source/license/dataset hashes, hashes the report and atomically persists JSON; repeatability, variance, provenance and persisted JSON tests; CI 171 passed and Ruff passed |
| 20 | Independent final review and fair comparison against selected public baselines | IN PROGRESS | Final static/security review and current-head CodeQL/Android emulator verification pending; fair comparison to a capable public language-model baseline is NOT VERIFIED because this prototype has no trained useful-scale model or curated public benchmark results |

## Verified progress

**95% — 19 of 20 Phase 2 packages complete.**

The current implementation is an early research prototype. It now includes a dependency-free reverse-mode scalar autodiff engine and a reference trainer that propagates gradients through embeddings, causal attention projections, attention output, feed-forward weights and the vocabulary projection. A deterministic tiny-corpus test verifies loss reduction and updates to embedding, attention and FFN weights; a full-model parameter gradient is checked against finite differences. This is a correctness milestone, not evidence of broad language competence or useful-scale training.

Current GitHub CI on commit `5d100a17dab87222c19135a360146ef36726a5e0`: 171 pytest tests passed; Ruff passed. CodeQL and Android emulator smoke for the current head are tracked separately. CodeQL and Android emulator checks are tracked separately and must not be inferred from Python CI. Android emulator smoke previously exposed `UnknownHostException` for `raw.githubusercontent.com`; the workflow now records remote sync as NOT VERIFIED when emulator DNS is unavailable, while independently checking the remote JSON from the runner and continuing UI/offline tests.

## Non-claims and release gates

- No claim that QUANTUM AI matches or exceeds GPT-6.
- No claim of superintelligence based on architecture, parameter count, or toy-corpus loss.
- Full end-to-end training, useful-scale licensed data, compute, held-out benchmarks, safety evaluation and independent reproduction are required before capability claims.
- A 99% ambition is a strategic design target, not a measured score.
