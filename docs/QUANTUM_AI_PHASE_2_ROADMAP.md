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
| 10 | Add learning-rate schedule and early stopping | NOT STARTED | Schedule and stopping-condition tests |
| 11 | Demonstrate controlled tiny-corpus overfit and held-out generalization measurement | NOT STARTED | Train/validation loss report; no cherry-picked claims |
| 12 | Add language-model perplexity and next-token accuracy evaluation | NOT STARTED | Known-answer metric tests |
| 13 | Add licensed corpus ingestion and source-level provenance audit | NOT STARTED | Dataset license/source audit; unresolved rights block training release |
| 14 | Add configurable temperature, top-k and top-p sampling | NOT STARTED | Deterministic sampling and probability validity tests |
| 15 | Add efficient inference context handling and KV cache | NOT STARTED | Cached/uncached logits equivalence tests |
| 16 | Add performance profiling and memory/throughput baselines | NOT STARTED | Reproducible hardware-labelled benchmark report |
| 17 | Add broad reasoning, coding, factuality and robustness benchmark harness | NOT STARTED | Fixed datasets, baselines and reproducible scoring |
| 18 | Add adversarial, prompt-injection and unsafe-tool-use evaluation | NOT STARTED | Threat-model-driven security tests |
| 19 | Run multiple-seed experiments and preserve full experiment provenance | NOT STARTED | Repeated runs, variance report and artifact hashes |
| 20 | Independent final review and fair comparison against selected public baselines | NOT STARTED | All applicable checks green; disclose compute/data limits; no superiority claim without evidence |

## Verified progress

**45% — 9 of 20 Phase 2 packages complete.**

The current implementation is an early research prototype. It now includes a dependency-free reverse-mode scalar autodiff engine and a reference trainer that propagates gradients through embeddings, causal attention projections, attention output, feed-forward weights and the vocabulary projection. A deterministic tiny-corpus test verifies loss reduction and updates to embedding, attention and FFN weights; a full-model parameter gradient is checked against finite differences. This is a correctness milestone, not evidence of broad language competence or useful-scale training.

Current GitHub CI on commit `dc40fd96b2bdeb794d44e43ae6dc681179c22a0a`: 117 pytest tests passed; Ruff passed. CodeQL and Android emulator checks are tracked separately and must not be inferred from Python CI. Android emulator smoke previously exposed `UnknownHostException` for `raw.githubusercontent.com`; the workflow now records remote sync as NOT VERIFIED when emulator DNS is unavailable, while independently checking the remote JSON from the runner and continuing UI/offline tests.

## Non-claims and release gates

- No claim that QUANTUM AI matches or exceeds GPT-6.
- No claim of superintelligence based on architecture, parameter count, or toy-corpus loss.
- Full end-to-end training, useful-scale licensed data, compute, held-out benchmarks, safety evaluation and independent reproduction are required before capability claims.
- A 99% ambition is a strategic design target, not a measured score.
