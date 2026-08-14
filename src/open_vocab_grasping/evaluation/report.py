from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


RATE_FIELDS = (
    "target_selection_correct",
    "detection_success",
    "candidate_generation_success",
    "ik_reachable",
    "grasp_success",
    "end_to_end_success",
)
TIME_FIELDS = (
    "detection_s",
    "grasp_generation_s",
    "graspnet_inference_s",
    "planning_s",
    "execution_s",
    "total_s",
)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(rows)}
    for field in RATE_FIELDS:
        values = [bool(row[field]) for row in rows if row.get(field) is not None]
        summary[f"{field}_rate"] = mean(values) if values else None
        summary[f"{field}_count"] = sum(values) if values else 0
    for field in TIME_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        summary[f"mean_{field}"] = mean(values) if values else None
    summary["failure_distribution"] = dict(
        sorted(Counter(str(row["failure_reason"]) for row in rows if row.get("failure_reason")).items())
    )
    return summary


def summarize_episodes(
    rows: list[dict[str, Any]], unavailable_modes: dict[str, str] | None = None
) -> dict[str, Any]:
    modes = sorted({str(row["mode"]) for row in rows})
    targets = sorted({str(row["target"]) for row in rows})
    return {
        "actual_run": True,
        "episode_count": len(rows),
        "targets": targets,
        "modes": {mode: _summarize_rows([row for row in rows if row["mode"] == mode]) for mode in modes},
        "by_target": {
            target: _summarize_rows([row for row in rows if row["target"] == target])
            for target in targets
        },
        "overall": _summarize_rows(rows),
        "unavailable_modes": unavailable_modes or {},
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Evaluation summary",
        "",
        f"Actual executed episodes: **{summary['episode_count']}**.",
        "",
        "## Per-mode metrics",
        "",
        "| Mode | Episodes | Detection | Target selection | Candidate generation | IK reachable | Grasp | End-to-end | Mean total s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in summary["modes"].items():
        def pct(field: str) -> str:
            value = values.get(field)
            return "n/a" if value is None else f"{100.0 * value:.1f}%"

        total = values.get("mean_total_s")
        lines.append(
            f"| {mode} | {values['episodes']} | {pct('detection_success_rate')} | "
            f"{pct('target_selection_correct_rate')} | {pct('candidate_generation_success_rate')} | "
            f"{pct('ik_reachable_rate')} | {pct('grasp_success_rate')} | "
            f"{pct('end_to_end_success_rate')} | {'n/a' if total is None else f'{total:.3f}'} |"
        )
    lines.extend(["", "## Failure distribution", ""])
    failures = summary["overall"]["failure_distribution"]
    if failures:
        lines.extend(f"- `{reason}`: {count}" for reason, count in failures.items())
    else:
        lines.append("- No failures in the executed episodes.")
    lines.extend(["", "## Unavailable modes", ""])
    unavailable = summary.get("unavailable_modes", {})
    if unavailable:
        lines.extend(f"- `{mode}`: {reason}" for mode, reason in unavailable.items())
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "All numbers above were computed from `episodes.csv`; unavailable modes are not assigned synthetic scores.",
            "",
        ]
    )
    return "\n".join(lines)
