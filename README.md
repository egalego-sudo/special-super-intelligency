# SPECIAL SUPER INTELLIGENCY

**Creator by Bittu Pandit**

An independent AI research and engineering project building a measurable, independently implemented intelligence stack.

## Mission

Develop an independently engineered AI system for reasoning, coding, scientific research, memory, multimodal processing, and permissioned tool use. The long-term ambition is to outperform leading AI systems in clearly defined domains through reproducible evaluations.

> The 99% target is an ambition for independence and capability, not a measured intelligence score. No GPT-6-class superiority is claimed.

## Current implemented foundations

- Deterministic UTF-8 byte tokenizer for Hindi, English, and Unicode text.
- Small trainable one-hidden-layer neural network with repository-owned weights.
- Versioned JSON model checkpoints with validation.
- Finite-difference check of the backpropagation gradient.
- Single-head causal self-attention research primitive.
- Byte-level next-token data preparation and toy regression training loop.
- Reproducible experiment configuration and regression evaluation metrics.
- JSONL dataset loader requiring text, source, and license metadata.
- Dataset metadata audit for duplicates, source labels, and license allowlists.
- Explicit in-memory knowledge store with source provenance and deletion.
- Permission-gated tool registry; elevated and high-risk tools require approval.
- Optional Docker sandbox runner with no-network, read-only, non-root, resource-limited settings and immutable image requirement.
- GitHub Actions CI, Dependabot, and CodeQL workflow.

## Current limitations

This is an early research prototype, not a trained foundation model or production AI assistant. The MLP and next-byte regressor are educational baselines; attention is not yet integrated into a trainable Transformer. There is no multimodal model, large-scale training pipeline, persistent encrypted memory, or autonomous multi-agent runtime. The Docker sandbox has a GitHub Actions integration test for non-root execution, read-only root filesystem, writable temporary storage, and disabled networking. This does not prove resistance to kernel/container escapes; highly hostile workloads still require a VM or dedicated sandbox.

## Verification

GitHub Actions CI has passed for the current codebase, including pytest and Ruff. CodeQL analysis is configured and its first run is being checked. Passing software tests does not establish broad intelligence or performance superiority.

## Development status

See [PROGRESS.md](PROGRESS.md) for the verified 5% work-package tracker, [docs/ROADMAP.md](docs/ROADMAP.md) for staged goals, and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system boundaries.

## Getting started

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

The CLI prototype is available as `ssi-core` after installation.

## License

No open-source license has been selected yet. Until one is added, the repository remains subject to default copyright rules; do not assume permission to reuse or redistribute its code.

## Creator

Creator by Bittu Pandit
