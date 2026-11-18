import pytest

from special_super_intelligency.core import Task, TaskRisk, create_plan


def test_create_plan_returns_verifiable_steps():
    plan = create_plan(Task("Build a small model prototype"))
    assert plan.objective == "Build a small model prototype"
    assert len(plan.steps) == 5
    assert plan.requires_human_approval is False


@pytest.mark.parametrize("risk", [TaskRisk.ELEVATED, TaskRisk.HIGH])
def test_elevated_risk_requires_human_approval(risk):
    plan = create_plan(Task("Deploy a service", risk=risk))
    assert plan.requires_human_approval is True


def test_empty_objective_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        create_plan(Task("   "))
