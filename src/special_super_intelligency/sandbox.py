"""Optional Docker-backed execution sandbox with restrictive defaults.

The caller must supply a locally available immutable image digest. No host paths,
Docker socket, secrets, or network are mounted into the execution container.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass


class SandboxError(RuntimeError):
    """Raised when isolated execution cannot be completed safely."""


class SandboxTimeout(SandboxError):
    """Raised when sandboxed code exceeds its wall-clock budget."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str


_WRAPPER = (
    "import sys\n"
    "source = sys.stdin.read()\n"
    "exec(compile(source, '<sandbox>', 'exec'), {'__name__': '__main__'})\n"
)


class DockerSandbox:
    """Run bounded Python snippets in a no-network, read-only Docker container."""

    def __init__(
        self,
        image: str,
        *,
        timeout_seconds: int = 5,
        max_code_bytes: int = 32_768,
        max_output_bytes: int = 65_536,
    ) -> None:
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Sandbox image must be pinned by an immutable sha256 digest.")
        if not 1 <= timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be between 1 and 30.")
        if not 1 <= max_code_bytes <= 262_144:
            raise ValueError("max_code_bytes must be between 1 and 262144.")
        if not 1 <= max_output_bytes <= 1_048_576:
            raise ValueError("max_output_bytes must be between 1 and 1048576.")
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_code_bytes = max_code_bytes
        self.max_output_bytes = max_output_bytes

    def build_command(self, container_name: str | None = None) -> list[str]:
        """Return fixed Docker arguments; user input is never interpolated as flags."""
        command = [
            "docker", "run", "--rm", "-i", "--pull=never",
            "--network=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--memory=128m", "--cpus=0.5",
            "--pids-limit=64", "--user=65534:65534",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            "--entrypoint=python", self.image, "-I", "-c", _WRAPPER,
        ]
        if container_name is not None:
            command[3:3] = ["--name", container_name]
        return command

    def run_python(self, source: str) -> SandboxResult:
        """Execute code with bounded input, output files, resources, and wall time."""
        if not isinstance(source, str):
            raise TypeError("source must be a string.")
        if len(source.encode("utf-8")) > self.max_code_bytes:
            raise ValueError("source exceeds the configured code-size limit.")
        if shutil.which("docker") is None:
            raise SandboxError("Docker CLI is unavailable; refusing host execution.")
        if os.name != "posix":
            raise SandboxError("The resource-limited Docker runner currently requires a POSIX host.")
        container_name = f"ssi-sandbox-{uuid.uuid4().hex}"

        def limit_output_file_size() -> None:
            import resource

            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (self.max_output_bytes, self.max_output_bytes),
            )
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                completed = subprocess.run(
                    self.build_command(container_name),
                    input=source,
                    text=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=self.timeout_seconds,
                    check=False,
                    shell=False,
                    preexec_fn=limit_output_file_size,
                )
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeout("Sandbox execution exceeded its time limit.") from exc
            except FileNotFoundError as exc:
                raise SandboxError("Docker CLI is unavailable; refusing host execution.") from exc
            finally:
                # Killing the Docker CLI on timeout does not guarantee daemon-side
                # container termination; explicitly remove the named container.
                try:
                    subprocess.run(
                        ["docker", "rm", "--force", container_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                        shell=False,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(self.max_output_bytes).decode("utf-8", errors="replace")
            stderr = stderr_file.read(self.max_output_bytes).decode("utf-8", errors="replace")
        return SandboxResult(completed.returncode, stdout, stderr)
