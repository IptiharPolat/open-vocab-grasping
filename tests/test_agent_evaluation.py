from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from open_vocab_grasping.agent.deepseek_client import PlannerResponse
from open_vocab_grasping.agent.evaluation import (
    expand_full_chain_cases,
    load_instruction_suite,
    run_agent_evaluation,
    summarize_agent_cases,
)
from open_vocab_grasping.agent.schemas import CANONICAL_PICK_STEPS, validate_grasp_plan


class _SuitePlanner:
    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
        target = "mug" if "杯子" in instruction else "bottle"
        plan = validate_grasp_plan(
            {
                "action": "pick",
                "target": target,
                "steps": list(CANONICAL_PICK_STEPS),
                "execution_mode": "open-vocab-simple",
                "explanation": "Test response.",
            },
            available_targets,
        )
        return PlannerResponse(
            "deepseek",
            "test-model",
            plan,
            {"messages": []},
            {"choices": []},
            0.2,
            {"total_tokens": 12},
        )


def test_load_instruction_suite_validates_cases(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "cases:\n  - id: one\n    language: zh\n    instruction: 抓杯子\n    expected_target: mug\n",
        encoding="utf-8",
    )
    assert load_instruction_suite(suite, ["mug"])[0]["id"] == "one"


def test_full_chain_expansion_uses_paired_target_seed_formula() -> None:
    cases = [
        {"id": "mug_a", "language": "zh", "instruction": "抓杯子", "expected_target": "mug"},
        {"id": "mug_b", "language": "en", "instruction": "pick mug", "expected_target": "mug"},
        {
            "id": "bottle_a",
            "language": "zh",
            "instruction": "抓瓶子",
            "expected_target": "bottle",
        },
    ]
    expanded = expand_full_chain_cases(cases, ["mug", "bottle"], 3, 10)
    assert [case["seed"] for case in expanded] == [10, 11, 12, 13, 14, 15]
    assert [case["source_case_id"] for case in expanded[:3]] == ["mug_a", "mug_b", "mug_a"]
    assert [case["expected_target"] for case in expanded] == [
        "mug",
        "mug",
        "mug",
        "bottle",
        "bottle",
        "bottle",
    ]


def test_agent_evaluation_writes_real_rows_and_summary(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "cases:\n"
        "  - {id: one, language: zh, instruction: 抓杯子, expected_target: mug}\n"
        "  - {id: two, language: en, instruction: pick bottle, expected_target: bottle}\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    configs = project / "configs"
    configs.mkdir(parents=True)
    config_path = configs / "agent.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    config: dict[str, Any] = {
        "_config_path": str(config_path),
        "output_root": "outputs",
        "scene": {"object_names": ["mug", "bottle"]},
        "agent": {"execution_mode": "open-vocab-simple"},
    }

    output, summary = run_agent_evaluation(
        config, suite, "deepseek", planner=_SuitePlanner()
    )

    assert summary["actual_run"] is True
    assert summary["successful_deepseek_response_count"] == 2
    assert summary["total_tokens"] == 24
    assert summary["overall"]["valid_plan_rate"] == 1.0
    assert summary["overall"]["target_accuracy"] == 1.0
    assert summary["overall"]["python_valid_rate"] == 1.0
    with (output / "cases.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert json.loads((output / "cases" / "one.json").read_text())["validated_plan"]["target"] == "mug"


def test_summary_keeps_wrong_target_in_denominator() -> None:
    rows = [
        {
            "language": "zh",
            "expected_target": "mug",
            "model": "model",
            "plan_valid": True,
            "target_correct": True,
            "python_valid": True,
            "planning_latency_s": 0.1,
            "total_tokens": 5,
            "robot_requested": False,
            "robot_executed": False,
            "robot_success": None,
            "error_type": None,
            "robot_failure_reason": None,
        },
        {
            "language": "zh",
            "expected_target": "bottle",
            "model": "model",
            "plan_valid": True,
            "target_correct": False,
            "python_valid": True,
            "planning_latency_s": 0.1,
            "total_tokens": 5,
            "robot_requested": True,
            "robot_executed": False,
            "robot_success": None,
            "error_type": None,
            "robot_failure_reason": "skipped_target_mismatch",
        },
    ]
    summary = summarize_agent_cases(rows, "deepseek")
    assert summary["overall"]["target_accuracy"] == 0.5
    assert summary["overall"]["robot_executed_count"] == 0
    assert summary["overall"]["full_chain_success_rate"] == 0.0
    assert summary["overall"]["robot_not_executed_count"] == 1
    assert summary["robot_failure_distribution"] == {"skipped_target_mismatch": 1}
