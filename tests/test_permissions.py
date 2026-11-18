import pytest

from special_super_intelligency.permissions import (
    ActionRisk,
    ApprovalRequired,
    ToolRegistry,
    ToolSpec,
)


def test_registered_low_risk_tool_can_run():
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", lambda args: args["text"]))
    assert registry.invoke("echo", {"text": "hello"}) == "hello"


def test_unregistered_tool_is_denied():
    registry = ToolRegistry()
    with pytest.raises(PermissionError):
        registry.invoke("shell", {})


def test_high_risk_tool_requires_human_approval():
    registry = ToolRegistry()
    registry.register(ToolSpec("publish_release", lambda _: "published", ActionRisk.HIGH))
    with pytest.raises(ApprovalRequired):
        registry.invoke("publish_release", {})
    assert registry.invoke("publish_release", {}, human_approved=True) == "published"


def test_duplicate_tool_names_are_rejected():
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", lambda _: "ok"))
    with pytest.raises(ValueError):
        registry.register(ToolSpec("echo", lambda _: "other"))


def test_invalid_argument_types_are_rejected():
    registry = ToolRegistry()
    registry.register(ToolSpec("echo", lambda _: "ok"))
    with pytest.raises(TypeError):
        registry.invoke("echo", {"bad": 1})  # type: ignore[dict-item]
