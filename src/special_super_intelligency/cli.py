"""Command-line interface for the starter planning core."""

import argparse

from .core import Task, TaskRisk, create_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="SPECIAL SUPER INTELLIGENCY planning prototype")
    parser.add_argument("objective", help="Task objective to plan")
    parser.add_argument(
        "--risk",
        choices=[risk.value for risk in TaskRisk],
        default=TaskRisk.LOW.value,
        help="Risk classification; elevated and high risk require human approval",
    )
    args = parser.parse_args()

    plan = create_plan(Task(objective=args.objective, risk=TaskRisk(args.risk)))
    print(f"Objective: {plan.objective}")
    for index, step in enumerate(plan.steps, start=1):
        print(f"{index}. {step}")
    print(f"Human approval required: {'yes' if plan.requires_human_approval else 'no'}")


if __name__ == "__main__":
    main()
