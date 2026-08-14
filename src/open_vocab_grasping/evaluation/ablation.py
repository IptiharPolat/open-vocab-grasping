from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from open_vocab_grasping.pipeline import run_graspnet_only, run_open_vocab_grasp


def run_filter_ablation(
    config: dict[str, Any], target: str, seed: int
) -> tuple[Path, dict[str, Any]]:
    if config.get("grasp", {}).get("generator") != "graspnet":
        raise ValueError("filter ablation requires a GraspNet configuration")
    project = Path(config["_config_path"]).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = project / config.get("output_root", "outputs") / f"{timestamp}_filter_ablation"
    output.mkdir(parents=True, exist_ok=False)
    variants: list[tuple[str, dict[str, Any], bool]] = []

    full = deepcopy(config)
    variants.append(("full", full, True))
    no_depth = deepcopy(config)
    no_depth["association"]["depth_tolerance_m"] = 10.0
    variants.append(("no_depth_consistency", no_depth, True))
    no_contact = deepcopy(config)
    no_contact["filters"]["minimum_contact_score"] = 0.0
    no_contact["filters"]["contact_occlusion_fallback_max_center_distance_m"] = 10.0
    variants.append(("no_contact_support", no_contact, True))
    no_clearance = deepcopy(config)
    no_clearance["filters"]["minimum_scene_clearance_m"] = 0.0
    variants.append(("no_scene_pointcloud_clearance", no_clearance, True))
    no_semantics = deepcopy(config)
    variants.append(("graspnet_only_no_text", no_semantics, False))

    rows: list[dict[str, Any]] = []
    for name, variant, semantic in variants:
        source, result = (
            run_open_vocab_grasp(variant, target, seed)
            if semantic
            else run_graspnet_only(variant, target, seed)
        )
        rows.append(
            {
                "variant": name,
                "target": target,
                "seed": seed,
                "semantic_selection": semantic,
                "success": bool(result.get("success", False)),
                "failure_reason": result.get("failure_reason"),
                "detection_count": result.get("detection_count"),
                "candidate_count": result.get("candidate_count"),
                "associated_count": result.get("filter_stage_counts", {}).get("associated"),
                "geometry_count": result.get("filter_stage_counts", {}).get("geometry"),
                "contact_count": result.get("filter_stage_counts", {}).get("contact_support"),
                "pointcloud_count": result.get("filter_stage_counts", {}).get("pointcloud_collision"),
                "ik_trajectory_count": result.get("filter_stage_counts", {}).get("ik_and_trajectory"),
                "selected_target_error_m": result.get("selected_candidate_target_center_error_m"),
                "lift_m": result.get("metrics", {}).get("lift_m"),
                "total_s": result.get("elapsed_s"),
                "source_output": str(source),
            }
        )

    summary = {
        "actual_run": True,
        "target": target,
        "seed": seed,
        "variant_count": len(rows),
        "successful_variants": [row["variant"] for row in rows if row["success"]],
        "rows": rows,
    }
    with (output / "ablations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Filter ablation",
        "",
        f"Actual target/seed: **{target} / {seed}**.",
        "",
        "| Variant | Semantic | Associated | Geometry | Contact | Cloud | IK/trajectory | Success | Lift m |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lift = row["lift_m"]
        lines.append(
            f"| {row['variant']} | {row['semantic_selection']} | {row['associated_count']} | "
            f"{row['geometry_count']} | {row['contact_count']} | {row['pointcloud_count']} | "
            f"{row['ik_trajectory_count']} | {row['success']} | "
            f"{'n/a' if lift is None else f'{float(lift):.6f}'} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, summary
