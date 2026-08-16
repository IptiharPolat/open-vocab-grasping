"""Reproducible multilingual planner and optional full-robot evaluation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from open_vocab_grasping.agent.codegen import compile_and_execute_plan
from open_vocab_grasping.agent.deepseek_client import PlannerResponse
from open_vocab_grasping.agent.runtime import (
    AgentExecution,
    Planner,
    build_planner,
    run_agent_instruction,
)
from open_vocab_grasping.config import save_config
from open_vocab_grasping.pipeline import write_environment_snapshot


FIELDS = (
    "case_id",
    "language",
    "instruction",
    "expected_target",
    "predicted_target",
    "provider",
    "model",
    "plan_valid",
    "target_correct",
    "python_valid",
    "planning_latency_s",
    "total_tokens",
    "robot_requested",
    "robot_executed",
    "robot_success",
    "robot_failure_reason",
    "seed",
    "source_output",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class _FixedResponsePlanner:
    response: PlannerResponse

    def plan(self, instruction: str, available_targets: list[str]) -> PlannerResponse:
        return self.response


def load_instruction_suite(path: str | Path, available_targets: list[str]) -> list[dict[str, str]]:
    suite_path = Path(path).expanduser().resolve()
    with suite_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"Instruction suite {suite_path} must contain a non-empty cases list")
    allowed_targets = {str(target) for target in available_targets}
    cases: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"Instruction suite case {index} must be an object")
        case_id = str(raw.get("id", "")).strip()
        instruction = str(raw.get("instruction", "")).strip()
        expected_target = str(raw.get("expected_target", "")).strip()
        language = str(raw.get("language", "")).strip().lower()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Instruction suite case {index} has a missing or duplicate id")
        if not instruction:
            raise ValueError(f"Instruction suite case {case_id!r} has an empty instruction")
        if expected_target not in allowed_targets:
            raise ValueError(
                f"Instruction suite case {case_id!r} expects {expected_target!r}, "
                f"outside scene targets {sorted(allowed_targets)}"
            )
        if language not in {"zh", "en"}:
            raise ValueError(f"Instruction suite case {case_id!r} language must be zh or en")
        seen_ids.add(case_id)
        cases.append(
            {
                "id": case_id,
                "instruction": instruction,
                "expected_target": expected_target,
                "language": language,
            }
        )
    return cases


def expand_full_chain_cases(
    cases: list[dict[str, str]],
    targets: list[str],
    episodes_per_target: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    """Pair every target with deterministic seeds and round-robin instructions."""
    if episodes_per_target <= 0:
        raise ValueError("episodes_per_target must be positive")
    grouped = {
        target: [case for case in cases if case["expected_target"] == target]
        for target in targets
    }
    missing = [target for target, target_cases in grouped.items() if not target_cases]
    if missing:
        raise ValueError(f"Instruction suite has no cases for targets: {missing}")
    expanded: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        target_cases = grouped[target]
        for episode_index in range(episodes_per_target):
            template = target_cases[episode_index % len(target_cases)]
            expanded.append(
                {
                    **template,
                    "id": f"{target}_full_{episode_index:02d}",
                    "source_case_id": template["id"],
                    "episode_index": episode_index,
                    "seed": seed_start + target_index * episodes_per_target + episode_index,
                }
            )
    return expanded


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["plan_valid"]]
    executed = [row for row in rows if row["robot_executed"]]
    requested = [row for row in rows if row["robot_requested"]]
    latencies = [float(row["planning_latency_s"]) for row in valid_rows]
    return {
        "cases": len(rows),
        "valid_plan_count": sum(bool(row["plan_valid"]) for row in rows),
        "valid_plan_rate": _rate(rows, "plan_valid"),
        "target_correct_count": sum(bool(row["target_correct"]) for row in rows),
        "target_accuracy": _rate(rows, "target_correct"),
        "target_accuracy_given_valid_plan": _rate(valid_rows, "target_correct"),
        "python_valid_count": sum(bool(row["python_valid"]) for row in rows),
        "python_valid_rate": _rate(rows, "python_valid"),
        "mean_planning_latency_s": mean(latencies) if latencies else None,
        "robot_requested_count": sum(bool(row["robot_requested"]) for row in rows),
        "robot_executed_count": len(executed),
        "robot_success_count": sum(bool(row["robot_success"]) for row in executed),
        "robot_success_rate": _rate(executed, "robot_success"),
        "full_chain_success_count": sum(bool(row.get("robot_success")) for row in requested),
        "full_chain_success_rate": (
            mean(bool(row.get("robot_success")) for row in requested) if requested else None
        ),
        "robot_not_executed_count": sum(not bool(row["robot_executed"]) for row in requested),
    }


def summarize_agent_cases(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    languages = sorted({str(row["language"]) for row in rows})
    targets = sorted({str(row["expected_target"]) for row in rows})
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    return {
        "actual_run": True,
        "requested_mode": mode,
        "successful_deepseek_response_count": sum(
            str(row.get("provider")) == "deepseek" for row in rows
        ),
        "case_count": len(rows),
        "models": dict(sorted(Counter(str(row["model"]) for row in rows if row.get("model")).items())),
        "total_tokens": total_tokens,
        "overall": _group_summary(rows),
        "by_language": {
            language: _group_summary([row for row in rows if row["language"] == language])
            for language in languages
        },
        "by_target": {
            target: _group_summary([row for row in rows if row["expected_target"] == target])
            for target in targets
        },
        "planning_failure_distribution": dict(
            sorted(Counter(str(row["error_type"]) for row in rows if row.get("error_type")).items())
        ),
        "robot_failure_distribution": dict(
            sorted(
                Counter(
                    str(row["robot_failure_reason"])
                    for row in rows
                    if row.get("robot_requested") and row.get("robot_failure_reason")
                ).items()
            )
        ),
    }


def agent_summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.1f}%"

    mean_latency = overall["mean_planning_latency_s"]
    latency_text = "n/a" if mean_latency is None else f"{mean_latency:.3f} s"

    lines = [
        "# Agent reliability evaluation",
        "",
        f"Actual instruction cases: **{summary['case_count']}**.",
        f"Requested planner mode: `{summary['requested_mode']}`.",
        f"Successful DeepSeek responses: **{summary['successful_deepseek_response_count']}**.",
        f"Total reported tokens: **{summary['total_tokens']}**.",
        "",
        "## Overall",
        "",
        f"- Schema-valid plans: {overall['valid_plan_count']}/{overall['cases']} "
        f"({pct(overall['valid_plan_rate'])})",
        f"- Correct targets: {overall['target_correct_count']}/{overall['cases']} "
        f"({pct(overall['target_accuracy'])})",
        f"- Valid generated Python: {overall['python_valid_count']}/{overall['cases']} "
        f"({pct(overall['python_valid_rate'])})",
        f"- Mean planning latency: {latency_text}",
        f"- Robot success among executed cases: {overall['robot_success_count']}/"
        f"{overall['robot_executed_count']} ({pct(overall['robot_success_rate'])})",
        f"- Full-chain success among requested cases: {overall['full_chain_success_count']}/"
        f"{overall['robot_requested_count']} ({pct(overall['full_chain_success_rate'])})",
        "",
        "## By target",
        "",
        "| Target | Cases | Valid plan | Target accuracy | Python valid | Full-chain success |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target, values in summary["by_target"].items():
        lines.append(
            f"| {target} | {values['cases']} | {pct(values['valid_plan_rate'])} | "
            f"{pct(values['target_accuracy'])} | {pct(values['python_valid_rate'])} | "
            f"{values['full_chain_success_count']}/{values['robot_requested_count']} "
            f"({pct(values['full_chain_success_rate'])}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Planner-only cases validate language understanding, schema compliance and generated-code safety. "
            "Rows with `robot_requested=true` form the full-chain denominator; planning, target or runtime "
            "failures that prevent execution still count as end-to-end failures.",
            "",
            "All values are computed from `cases.csv`; failed cases remain in the denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def run_agent_evaluation(
    config: dict[str, Any],
    suite_path: str | Path,
    mode: str,
    robot_cases_per_target: int = 0,
    seed_start: int = 0,
    full_episodes_per_target: int = 0,
    *,
    planner: Planner | None = None,
) -> tuple[Path, dict[str, Any]]:
    if robot_cases_per_target < 0:
        raise ValueError("robot_cases_per_target must be non-negative")
    if full_episodes_per_target < 0:
        raise ValueError("full_episodes_per_target must be non-negative")
    if robot_cases_per_target and full_episodes_per_target:
        raise ValueError(
            "robot_cases_per_target and full_episodes_per_target are mutually exclusive"
        )
    targets = [str(target) for target in config["scene"]["object_names"]]
    suite_cases = load_instruction_suite(suite_path, targets)
    cases: list[dict[str, Any]] = (
        expand_full_chain_cases(
            suite_cases, targets, full_episodes_per_target, seed_start
        )
        if full_episodes_per_target
        else list(suite_cases)
    )
    selected_planner = planner or build_planner(mode, config)
    project = Path(config["_config_path"]).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = project / config.get("output_root", "outputs") / f"{timestamp}_agent_evaluation"
    (output / "cases").mkdir(parents=True, exist_ok=False)

    selected_for_robot: set[str] = (
        {str(case["id"]) for case in cases} if full_episodes_per_target else set()
    )
    selected_counts: Counter[str] = Counter()
    if not full_episodes_per_target:
        for case in cases:
            target = case["expected_target"]
            if selected_counts[target] < robot_cases_per_target:
                selected_for_robot.add(case["id"])
                selected_counts[target] += 1

    rows: list[dict[str, Any]] = []
    robot_seed_index = 0
    for case in cases:
        row: dict[str, Any] = {
            "case_id": case["id"],
            "language": case["language"],
            "instruction": case["instruction"],
            "expected_target": case["expected_target"],
            "predicted_target": None,
            "provider": None,
            "model": None,
            "plan_valid": False,
            "target_correct": False,
            "python_valid": False,
            "planning_latency_s": None,
            "total_tokens": 0,
            "robot_requested": case["id"] in selected_for_robot,
            "robot_executed": False,
            "robot_success": None,
            "robot_failure_reason": None,
            "seed": case.get("seed"),
            "source_output": None,
            "error_type": None,
            "error_message": None,
        }
        case_audit: dict[str, Any] = {"case": case, "requested_mode": mode}
        try:
            response = selected_planner.plan(case["instruction"], targets)
            row.update(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "plan_valid": True,
                    "predicted_target": response.plan["target"],
                    "target_correct": response.plan["target"] == case["expected_target"],
                    "planning_latency_s": response.latency_s,
                    "total_tokens": int(response.usage.get("total_tokens", 0)),
                }
            )
            source, trace = compile_and_execute_plan(response.plan)
            row["python_valid"] = True
            case_audit.update(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "sanitized_request": response.request,
                    "response": response.response,
                    "validated_plan": response.plan,
                    "generated_python": source,
                    "validation_trace": trace,
                    "usage": response.usage,
                    "latency_s": response.latency_s,
                }
            )
            if row["robot_requested"] and row["target_correct"]:
                if case.get("seed") is not None:
                    seed = int(case["seed"])
                else:
                    seed = seed_start + robot_seed_index
                    robot_seed_index += 1
                row["robot_executed"] = True
                execution: AgentExecution = run_agent_instruction(
                    config,
                    case["instruction"],
                    seed,
                    mode,
                    planner=_FixedResponsePlanner(response),
                )
                row.update(
                    {
                        "robot_success": execution.success,
                        "robot_failure_reason": execution.robot_result.get("failure_reason"),
                        "seed": seed,
                        "source_output": str(execution.output),
                    }
                )
                case_audit["robot_execution"] = execution.to_dict()
            elif row["robot_requested"]:
                row["robot_failure_reason"] = "skipped_target_mismatch"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            if row["robot_requested"] and row["robot_failure_reason"] is None:
                row["robot_failure_reason"] = (
                    "robot_runtime_exception" if row["robot_executed"] else "planning_or_validation_failed"
                )
            case_audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
        rows.append(row)
        (output / "cases" / f"{case['id']}.json").write_text(
            json.dumps(case_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = summarize_agent_cases(rows, mode)
    summary["suite_path"] = str(Path(suite_path).expanduser().resolve())
    summary["robot_cases_per_target"] = robot_cases_per_target
    summary["full_episodes_per_target"] = full_episodes_per_target
    summary["seed_start"] = seed_start
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.md").write_text(agent_summary_markdown(summary), encoding="utf-8")
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    return output, summary
