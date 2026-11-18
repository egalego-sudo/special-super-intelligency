# QUANTUM AI — Phase 2 Security Threat Model

**Scope:** research-model evaluation, permissioned tool invocation and the optional Docker code-execution boundary. This document records engineering controls; it is not a security certification.

## Assets

- Host files, environment variables, credentials and Docker daemon.
- Repository and training/checkpoint artifacts.
- User approval for high-impact actions.
- Benchmark integrity, provenance metadata and evaluation results.

## Threats and controls

| Threat | Required control | Regression evidence |
|---|---|---|
| Prompt injection claims that a user approved a destructive action | Approval is a trusted caller argument, never inferred from model text or tool arguments | test_prompt_injection_text_cannot_authorize_high_risk_tool; test_model_supplied_approval_argument_is_not_an_approval_capability |
| Unregistered or arbitrary tool invocation | Explicit registry allowlist; unknown tools denied | test_unregistered_tool_is_denied |
| Tool name or container identifier flag injection | Strict identifier validation; argument values are not shell fragments | test_container_name_cannot_inject_docker_flags |
| Mutable or option-like sandbox image reference | Require immutable SHA-256 digest and a constrained image-reference syntax | test_rejects_malformed_or_option_like_image_references |
| Code escapes container through network, writable host mounts or Linux capabilities | --network=none, read-only root, no host mounts/socket, drop all capabilities, no-new-privileges, non-root UID, memory/CPU/PID limits and bounded tmpfs | test_sandbox_command_enforces_container_boundaries; test_sandbox_command_has_no_host_mounts_or_network_capabilities |
| Infinite loop or oversized source/output | Wall-clock timeout, code-size cap, bounded output and forced named-container cleanup | test_timeout_is_reported; test_rejects_oversized_source; test_runs_without_shell_and_returns_bounded_output |
| Sandbox unavailable and code falls back to host execution | Fail closed; never execute submitted code on host | test_missing_docker_refuses_to_run_on_host |
| Benchmark result inflated by invented citations or mismatched baselines | Evidence IDs must be declared; citation precision/coverage and identical dataset/category/metric checks | tests/test_benchmarks.py |

## Known limitations / NOT VERIFIED

- The Python registry is an in-process policy boundary, not a hardened multi-tenant authorization service.
- The approval boolean is safe only when set exclusively by a trusted UI/service layer; untrusted model output must never populate it.
- Docker daemon/kernel vulnerabilities and host misconfiguration remain outside the container command's guarantees.
- The optional Docker sandbox has not been exercised against a real adversarial container on a dedicated hardened host in this phase.
- No formal penetration test, fuzz campaign, kernel escape assessment or independent security audit has been completed.
- Model-level prompt-injection resistance is not measurable until a capable model/tool orchestration loop exists. Current tests validate the permission boundary, not the intelligence of the model.

## Release posture

High-risk publication, deletion, production deployment and irreversible actions remain approval-gated. Security test success does not authorize those actions.
