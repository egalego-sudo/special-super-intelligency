# QUANTUM AI Capability Benchmark Protocol

**Creator by Bittu Pandit**

The benchmark harness in `special_super_intelligency.benchmarks` scores four distinct task families:

- **Reasoning:** normalized exact-answer accuracy on fixed prompts.
- **Coding:** expected-output matching. Candidate-generated code is never executed by this scorer.
- **Factuality:** supported/unsupported label accuracy plus citation coverage.
- **Robustness:** paired-prompt consistency checks, with each pair explicitly labelled as invariant or expected to change.

Every benchmark manifest must identify its name, version, split, source and license. Scoring requires an explicit operator-approved license allowlist. The report includes a deterministic SHA-256 hash of the manifest and cases. Baseline comparison is rejected unless the dataset hashes match exactly.

## Evidence rules

1. Preserve the exact dataset version, source, license review, split and hash.
2. Keep candidate and baseline settings, decoding parameters and compute environment.
3. Report per-category metrics; do not collapse factuality or robustness into a single answer-accuracy number.
4. Report missing citations separately from supported-label correctness.
5. Do not execute candidate code in the scoring process. Use a separately isolated, reviewed execution harness if executable code tests are later required.
6. Synthetic unit fixtures validate the scoring implementation only. They are not a representative language, reasoning, coding or factuality benchmark.
7. No GPT-6 or other model superiority claim is permitted without a fixed, licensed, representative benchmark suite, matched settings, repeated runs, statistical uncertainty and independent reproduction.

The current implementation is a benchmark framework and scoring contract. Broad real-world capability remains **NOT VERIFIED** until licensed benchmark datasets and independently reproduced model runs are added.
