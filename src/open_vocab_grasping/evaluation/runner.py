from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from open_vocab_grasping.config import save_config
from open_vocab_grasping.evaluation.report import summarize_episodes, summary_markdown
from open_vocab_grasping.pipeline import (
    run_cpu_smoke,
    run_graspnet_only,
    run_open_vocab_grasp,
    write_environment_snapshot,
)


FIELDS = (
    "episode_id", "mode", "target", "seed", "success", "failure_reason",
    "detection_success", "target_selection_correct", "candidate_generation_success",
    "candidate_count", "accepted_count", "ik_reachable", "grasp_success",
    "end_to_end_success", "detection_s", "grasp_generation_s", "graspnet_inference_s", "planning_s",
    "execution_s", "total_s", "lift_m", "tool_object_distance_m", "source_output",
)


def _episode_row(mode: str, target: str, seed: int, output: Path, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    oracle = mode == "oracle-perception"
    semantic_free = mode == "graspnet-only"
    success = bool(result.get("success", False))
    return {
        "episode_id": f"{mode}_{target}_seed{seed}",
        "mode": mode,
        "target": target,
        "seed": seed,
        "success": success,
        "failure_reason": result.get("failure_reason"),
        "detection_success": True if oracle else (None if semantic_free else bool(result.get("detection_success", False))),
        "target_selection_correct": True if oracle else (None if semantic_free else bool(result.get("target_selection_correct", False))),
        "candidate_generation_success": bool(result.get("candidate_count", 0) > 0),
        "candidate_count": int(result.get("candidate_count", 0)),
        "accepted_count": int(result.get("accepted_count", 0)),
        "ik_reachable": bool(result.get("accepted_count", 0) > 0) if oracle else bool(result.get("ik_reachable", False)),
        "grasp_success": success,
        "end_to_end_success": success,
        "detection_s": 0.0 if oracle else result.get("detector_wall_s"),
        "grasp_generation_s": result.get("grasp_generation_s"),
        "graspnet_inference_s": result.get("graspnet_inference_s"),
        "planning_s": result.get("planning_s"),
        "execution_s": result.get("execution_s"),
        "total_s": result.get("elapsed_s"),
        "lift_m": metrics.get("lift_m"),
        "tool_object_distance_m": metrics.get("tool_object_distance_m"),
        "source_output": str(output),
    }


def _copy_artifacts(evaluation: Path, row: dict[str, Any], source: Path) -> None:
    stem = str(row["episode_id"])
    for filename in ("demo.mp4", "demo.gif"):
        artifact = source / filename
        if artifact.exists():
            shutil.copy2(artifact, evaluation / "videos" / f"{stem}{artifact.suffix}")
    for filename in ("detections.png", "raw_predictions.json"):
        artifact = source / filename
        if artifact.exists():
            shutil.copy2(artifact, evaluation / "detections" / f"{stem}_{filename}")
    for filename in ("filtered_candidates_3d.png", "pointcloud.ply"):
        artifact = source / filename
        if artifact.exists():
            shutil.copy2(artifact, evaluation / "pointclouds" / f"{stem}_{filename}")
    if not row["success"]:
        failure = evaluation / "failure_cases" / stem
        failure.mkdir(parents=True, exist_ok=True)
        for filename in (
            "result.json", "candidates.json", "detections.png", "filtered_candidates_2d.png",
            "filtered_candidates_3d.png", "rgb.png", "demo.gif",
        ):
            artifact = source / filename
            if artifact.exists():
                shutil.copy2(artifact, failure / filename)


def run_evaluation(
    config: dict[str, Any], targets: list[str], episodes_per_target: int, modes: list[str]
) -> tuple[Path, dict[str, Any]]:
    if episodes_per_target <= 0:
        raise ValueError("episodes_per_target must be positive")
    supported = {"oracle-perception", "graspnet-only", "open-vocab-simple", "open-vocab-graspnet"}
    unknown = sorted(set(modes) - supported)
    if unknown:
        raise ValueError(f"Unsupported executable evaluation modes: {unknown}")
    project = Path(config["_config_path"]).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    evaluation = project / config.get("output_root", "outputs") / f"{timestamp}_evaluation"
    for directory in ("videos", "detections", "pointclouds", "failure_cases"):
        (evaluation / directory).mkdir(parents=True, exist_ok=True)
    seed_start = int(config.get("evaluation", {}).get("seed_start", config.get("seed", 0)))
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for target_index, target in enumerate(targets):
            for episode_index in range(episodes_per_target):
                seed = seed_start + target_index * episodes_per_target + episode_index
                if mode == "oracle-perception":
                    source, result = run_cpu_smoke(config, target, seed)
                elif mode == "graspnet-only":
                    episode_config = deepcopy(config)
                    episode_config["grasp"]["generator"] = "graspnet"
                    if "graspnet" not in episode_config:
                        raise ValueError(
                            "graspnet-only evaluation requires the graspnet service config; "
                            "extend configs/graspnet.yaml"
                        )
                    source, result = run_graspnet_only(episode_config, target, seed)
                else:
                    episode_config = deepcopy(config)
                    episode_config["grasp"]["generator"] = (
                        "graspnet" if mode == "open-vocab-graspnet" else "geometric_baseline"
                    )
                    if mode == "open-vocab-graspnet" and "graspnet" not in episode_config:
                        raise ValueError(
                            "open-vocab-graspnet evaluation requires the graspnet service config; "
                            "extend configs/graspnet.yaml"
                        )
                    source, result = run_open_vocab_grasp(episode_config, target, seed)
                row = _episode_row(mode, target, seed, source, result)
                rows.append(row)
                _copy_artifacts(evaluation, row, source)
    unavailable: dict[str, str] = {}
    summary = summarize_episodes(rows, unavailable)
    summary["episodes_per_target_per_mode"] = episodes_per_target
    summary["seed_start"] = seed_start
    summary["planned_episode_count"] = len(targets) * episodes_per_target * len(modes)
    with (evaluation / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (evaluation / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (evaluation / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    save_config(config, evaluation / "config.yaml")
    write_environment_snapshot(evaluation / "environment.json")
    return evaluation, summary
