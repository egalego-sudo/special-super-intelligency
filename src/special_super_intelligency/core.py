"""Small deterministic orchestration core; not a trained language model."""

from dataclasses import dataclass
from enum import Enum


class TaskRisk(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class Task:
    objective: str
    risk: TaskRisk = TaskRisk.LOW


@dataclass(frozen=True)
class TaskPlan:
    objective: str
    steps: tuple[str, ...]
    requires_human_approval: bool


def create_plan(task: Task) -> TaskPlan:
    """Create a transparent starter plan without executing external actions."""
    objective = task.objective.strip()
    if not objective:
        raise ValueError("Task objective must not be empty.")

    steps = (
        "Clarify the desired outcome and acceptance criteria.",
        "Gather relevant evidence and identify assumptions.",
        "Propose a solution and identify possible failure modes.",
        "Verify the result with tests or independent checks.",
        "Report results, uncertainty, and remaining work.",
    )
    return TaskPlan(
        objective=objective,
        steps=steps,
        requires_human_approval=task.risk in {TaskRisk.ELEVATED, TaskRisk.HIGH},
    )
