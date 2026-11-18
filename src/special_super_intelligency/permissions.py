"""Permissioned in-process tool registry; no arbitrary shell or network execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ActionRisk(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[[dict[str, str]], str]
    risk: ActionRisk = ActionRisk.LOW


class ApprovalRequired(PermissionError):
    """Raised when an action requires explicit human approval."""


class ToolRegistry:
    """Allow only explicitly registered tools and gate elevated/high-risk actions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or not spec.name.replace("_", "").isalnum():
            raise ValueError("Tool names must contain only letters, digits, and underscores.")
        if spec.name in self._tools:
            raise ValueError(f"Tool {spec.name!r} is already registered.")
        self._tools[spec.name] = spec

    def invoke(
        self, name: str, arguments: dict[str, str], *, human_approved: bool = False
    ) -> str:
        if name not in self._tools:
            raise PermissionError(f"Tool {name!r} is not registered.")
        spec = self._tools[name]
        if spec.risk is not ActionRisk.LOW and not human_approved:
            raise ApprovalRequired(f"Tool {name!r} requires explicit human approval.")
        if not isinstance(arguments, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in arguments.items()
        ):
            raise TypeError("Tool arguments must be a string-to-string dictionary.")
        return spec.handler(arguments)
