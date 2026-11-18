# Development Roadmap

**Project:** SPECIAL SUPER INTELLIGENCY  
**Creator:** Creator by Bittu Pandit

This roadmap prioritizes verifiable foundations over unsupported claims.

## Phase 0 — Repository foundation
- [x] Expand project README and current status
- [x] Add architecture and contribution documentation
- [ ] Select an open-source license and project governance rules
- [x] Add CI checks, Dependabot, and CodeQL configuration

## Phase 1 — Requirements and baselines
- [x] Define an initial architecture and staged work plan
- [ ] Define intended users, deployment targets, and hardware/resource budgets
- [ ] Define capability domains and acceptance criteria for real reasoning tasks
- [x] Add a transparent toy evaluation harness
- [ ] Select public evaluation datasets and document contamination controls
- [ ] Record comparable external baseline results

## Phase 2 — Independent core prototype
- [x] Implement an inspectable trainable neural-network baseline
- [x] Define a reversible byte tokenizer and data format
- [x] Add deterministic configuration and reproducible toy runs
- [x] Add model checkpoint save/load and validation
- [x] Add a single-head causal attention research primitive
- [ ] Integrate trainable attention into a complete Transformer block
- [ ] Add numerical tests for attention and full model gradients

## Phase 3 — Reasoning, tools, and memory
- [x] Build structured task planning and permissioned tool interfaces
- [x] Add explicit in-memory knowledge retrieval with provenance
- [ ] Add durable, encrypted, user-controlled memory and retention controls
- [ ] Add verifier/critic workflows with evidence tracking
- [x] Add an optional Docker sandbox runner with restrictive defaults and immutable image pinning
- [x] Verify Docker integration in GitHub Actions for non-root, read-only root, no-network, and temporary storage boundaries
- [ ] Assess VM/dedicated sandbox isolation for highly hostile workloads

## Phase 4 — Training and improvement
- [x] Build basic dataset validation and metadata audit
- [x] Establish a tiny next-byte training experiment
- [ ] Complete source-term and legal review for any training corpus
- [ ] Build scalable training, checkpointing, and resume workflows
- [ ] Track compute, cost, energy, and model quality
- [ ] Use isolated experiments and regression gates

## Phase 5 — Evaluation and scaling
- [x] Add basic regression metrics and CI regression checks
- [ ] Evaluate reasoning, coding, math, factuality, safety, latency, and cost
- [ ] Compare against external systems under equivalent conditions
- [ ] Publish reproducible results and known limitations
- [ ] Scale only when evidence and resources justify it

## Release gates

No production release based solely on self-reported scores. Require passing tests, security review, reproducible evaluation, documented limitations, a selected license, and human approval.
