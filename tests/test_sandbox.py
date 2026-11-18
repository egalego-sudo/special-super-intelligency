import subprocess

import pytest

from special_super_intelligency.sandbox import (
    DockerSandbox,
    SandboxError,
    SandboxTimeout,
)


IMAGE = "python:3.11-slim@sha256:" + "a" * 64


def test_requires_immutable_image_digest():
    with pytest.raises(ValueError):
        DockerSandbox("python:3.11-slim")


def test_build_command_enforces_container_boundaries():
    command = DockerSandbox(IMAGE).build_command()
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--memory=128m" in command
    assert "--pids-limit=64" in command
    assert "--pull=never" in command
    assert not any("docker.sock" in item for item in command)


def test_rejects_oversized_source(monkeypatch):
    sandbox = DockerSandbox(IMAGE, max_code_bytes=4)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")
    with pytest.raises(ValueError):
        sandbox.run_python("print('too long')")


def test_runs_without_shell_and_returns_bounded_output(monkeypatch):
    sandbox = DockerSandbox(IMAGE)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")

    def fake_run(command, **kwargs):
        assert command[0:2] == ["docker", "run"]
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 5
        kwargs["stdout"].write(b"safe output")
        kwargs["stderr"].write(b"")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", fake_run)
    result = sandbox.run_python("print('safe output')")
    assert result.exit_code == 0
    assert result.stdout == "safe output"


def test_timeout_is_reported(monkeypatch):
    sandbox = DockerSandbox(IMAGE)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/docker")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("subprocess.run", fake_run)
    with pytest.raises(SandboxTimeout):
        sandbox.run_python("while True: pass")


def test_missing_docker_refuses_to_run_on_host(monkeypatch):
    sandbox = DockerSandbox(IMAGE)
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(SandboxError, match="refusing host execution"):
        sandbox.run_python("print('hello')")
