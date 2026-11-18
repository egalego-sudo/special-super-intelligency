# Docker Sandbox Boundary

**Creator by Bittu Pandit**

The optional `DockerSandbox` requires a locally available image pinned by SHA-256 digest. It refuses to fall back to host execution when Docker is missing.

The runner uses:
- no network (`--network=none`)
- read-only container root
- no Linux capabilities and no-new-privileges
- non-root UID/GID
- memory, CPU, process-count, wall-time, source-size, and output-file limits
- no host mounts, Docker socket, environment secrets, or automatic image pulls
- no shell interpolation

## Verification boundary

Unit tests verify command construction, input limits, timeout handling, and fail-closed behavior using a mocked Docker CLI. They do not prove the Docker daemon, host kernel, image contents, or real-world isolation. Before production use, run a dedicated integration test on a hardened Linux host with a reviewed immutable image digest. Container isolation is not a substitute for a VM or dedicated sandbox for highly hostile workloads.
