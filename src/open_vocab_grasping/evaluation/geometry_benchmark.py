from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from open_vocab_grasping.config import save_config
from open_vocab_grasping.geometry.projection import backproject_pixels, project_points
from open_vocab_grasping.geometry.transforms import invert_transform, transform_points
from open_vocab_grasping.grasping.geometric_baseline import GeometricTopDownGenerator
from open_vocab_grasping.pipeline import write_environment_snapshot
from open_vocab_grasping.simulation.camera import segmentation_body_ids
from open_vocab_grasping.simulation.world import SimulationWorld


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def run_geometry_benchmark(
    config: dict[str, Any], episodes: int, seed_start: int | None = None
) -> tuple[Path, dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    first_seed = int(config.get("seed", 0) if seed_start is None else seed_start)
    project = Path(config["_config_path"]).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = project / config.get("output_root", "outputs") / f"{timestamp}_geometry_benchmark"
    output.mkdir(parents=True, exist_ok=False)

    episode_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    generator = GeometricTopDownGenerator()
    rng = np.random.default_rng(first_seed)
    for episode_index in range(episodes):
        seed = first_seed + episode_index
        with SimulationWorld.create(config, seed) as world:
            observation = world.camera.capture()
            height, width = observation.depth_m.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            pixels = np.column_stack((u.ravel(), v.ravel())).astype(np.float64)
            camera_points = backproject_pixels(
                pixels, observation.depth_m.ravel(), observation.intrinsic
            )
            world_points = transform_points(observation.T_world_camera, camera_points)
            body_ids = segmentation_body_ids(observation.segmentation).ravel()
            scene_indices = np.flatnonzero(body_ids >= 0)
            sample_count = min(512, len(scene_indices))
            sampled = rng.choice(scene_indices, size=sample_count, replace=False)
            recovered_pixels, _ = project_points(camera_points[sampled], observation.intrinsic)
            projection_error = np.linalg.norm(recovered_pixels - pixels[sampled], axis=1)
            recovered_camera = transform_points(
                invert_transform(observation.T_world_camera), world_points[sampled]
            )
            transform_error = np.linalg.norm(recovered_camera - camera_points[sampled], axis=1)

            table_points = world_points[body_ids == world.table_id]
            table_median_z = float(np.median(table_points[:, 2]))
            table_error = abs(table_median_z - float(config["scene"]["table_top_z"]))
            episode_rows.append(
                {
                    "seed": seed,
                    "projection_mean_px": float(np.mean(projection_error)),
                    "projection_max_px": float(np.max(projection_error)),
                    "transform_mean_m": float(np.mean(transform_error)),
                    "transform_max_m": float(np.max(transform_error)),
                    "table_median_z_m": table_median_z,
                    "table_error_m": table_error,
                }
            )

            for name, item in world.objects.items():
                visible_points = world_points[body_ids == item.body_id]
                truth_min, truth_max = world.object_aabb(name)
                truth_center = (truth_min + truth_max) / 2.0
                visible_median = np.median(visible_points, axis=0)
                generated = generator.generate_from_points(visible_points)
                robust_center = generated[0].center if generated else visible_median
                object_rows.append(
                    {
                        "seed": seed,
                        "object": name,
                        "visible_points": len(visible_points),
                        "visible_median_center_error_m": float(
                            np.linalg.norm(visible_median - truth_center)
                        ),
                        "robust_geometry_center_error_m": float(
                            np.linalg.norm(robust_center - truth_center)
                        ),
                    }
                )

    projection_max = [float(row["projection_max_px"]) for row in episode_rows]
    transform_max = [float(row["transform_max_m"]) for row in episode_rows]
    table_errors = [float(row["table_error_m"]) for row in episode_rows]
    visible_errors = [float(row["visible_median_center_error_m"]) for row in object_rows]
    robust_errors = [float(row["robust_geometry_center_error_m"]) for row in object_rows]
    thresholds = {
        "projection_max_px": 1e-9,
        "transform_max_m": 1e-9,
        "table_max_error_m": 0.005,
        "robust_object_mean_error_m": 0.03,
        "robust_object_max_error_m": 0.06,
    }
    summary = {
        "actual_run": True,
        "episodes": episodes,
        "seed_start": first_seed,
        "object_samples": len(object_rows),
        "projection_max_px": _aggregate(projection_max),
        "transform_max_m": _aggregate(transform_max),
        "table_error_m": _aggregate(table_errors),
        "visible_median_object_center_error_m": _aggregate(visible_errors),
        "robust_geometry_object_center_error_m": _aggregate(robust_errors),
        "thresholds": thresholds,
    }
    robust_summary = summary["robust_geometry_object_center_error_m"]
    summary["passed"] = bool(
        summary["projection_max_px"]["maximum"] <= thresholds["projection_max_px"]
        and summary["transform_max_m"]["maximum"] <= thresholds["transform_max_m"]
        and summary["table_error_m"]["maximum"] <= thresholds["table_max_error_m"]
        and robust_summary["mean"] <= thresholds["robust_object_mean_error_m"]
        and robust_summary["maximum"] <= thresholds["robust_object_max_error_m"]
    )

    for filename, rows in (("episodes.csv", episode_rows), ("objects.csv", object_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = [
        "# Geometry benchmark",
        "",
        f"Actual fixed-seed episodes: **{episodes}**; object samples: **{len(object_rows)}**.",
        "",
        f"Overall threshold result: **{'PASS' if summary['passed'] else 'FAIL'}**.",
        "",
        "| Metric | Mean | P95 | Maximum |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("Projection round trip (px)", "projection_max_px"),
        ("Camera/world round trip (m)", "transform_max_m"),
        ("Table height error (m)", "table_error_m"),
        ("Visible median object center error (m)", "visible_median_object_center_error_m"),
        ("Robust geometry object center error (m)", "robust_geometry_object_center_error_m"),
    ):
        values = summary[key]
        markdown.append(
            f"| {label} | {values['mean']:.9g} | {values['p95']:.9g} | {values['maximum']:.9g} |"
        )
    (output / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    return output, summary
