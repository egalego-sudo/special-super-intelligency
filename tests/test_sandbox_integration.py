import os

import pytest

from special_super_intelligency.sandbox import DockerSandbox


def test_real_docker_sandbox_is_non_root_read_only_and_no_network():
    image = os.environ.get("SSI_SANDBOX_IMAGE")
    if not image:
        pytest.skip("SSI_SANDBOX_IMAGE is not configured for this environment.")

    source = """
import os
import socket

assert os.getuid() != 0, "container process must not run as root"
try:
    with open("/sandbox-write-check", "w", encoding="utf-8") as handle:
        handle.write("not allowed")
except OSError:
    pass
else:
    raise AssertionError("container root filesystem must be read-only")

with open("/tmp/sandbox-write-check", "w", encoding="utf-8") as handle:
    handle.write("temporary storage is available")

try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
except OSError:
    print("sandbox-ok")
else:
    raise AssertionError("container network must be disabled")
"""
    result = DockerSandbox(image, timeout_seconds=10).run_python(source)
    assert result.exit_code == 0, result.stderr
    assert "sandbox-ok" in result.stdout
