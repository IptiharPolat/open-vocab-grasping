from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import imageio.v3 as iio
import numpy as np

from open_vocab_grasping.config import save_config
from open_vocab_grasping.evaluation.metrics import detection_metrics
from open_vocab_grasping.geometry.projection import backproject_pixels
from open_vocab_grasping.geometry.transforms import invert_transform, transform_points
from open_vocab_grasping.grasping.association import Detection, associate_candidates_with_records
from open_vocab_grasping.grasping.geometric_baseline import GeometricTopDownGenerator, generate_scene_clusters
from open_vocab_grasping.grasping.graspnet_client import (
    GraspNetFileClient,
    camera_grasps_to_world_tool,
)
from open_vocab_grasping.grasping.ranking import rank_candidates
from open_vocab_grasping.perception.pointcloud import crop_workspace, rgbd_to_pointcloud, write_ply
from open_vocab_grasping.perception.yolo_world import YOLOWorldDetector, save_detections
from open_vocab_grasping.planning.executor import GraspExecutor
from open_vocab_grasping.planning.filtering import filter_grasp_candidates
from open_vocab_grasping.simulation.camera import segmentation_body_ids
from open_vocab_grasping.simulation.world import SimulationWorld
from open_vocab_grasping.visualization.overlay import save_association_overlay, save_detection_overlay, save_rgb
from open_vocab_grasping.visualization.pointcloud_vis import save_candidate_topdown, save_topdown_preview


def make_run_dir(config: dict[str, Any], kind: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(config["_config_path"]).parent.parent / config.get("output_root", "outputs")
    output = root / f"{timestamp}_{kind}_seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_environment_snapshot(path: Path) -> None:
    versions: dict[str, str] = {}
    for name in ("numpy", "pybullet", "open3d", "yaml", "torch", "torchvision", "ultralytics"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "installed"))
    path.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "executable": sys.executable,
                "packages": versions,
                "cuda_available": bool(versions.get("torch")) and __import__("torch").cuda.is_available(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def oracle_detection(world: SimulationWorld, observation: Any, target: str) -> Detection:
    body_ids = segmentation_body_ids(observation.segmentation)
    target_id = world.objects[target].body_id
    rows, columns = np.nonzero(body_ids == target_id)
    if len(rows) == 0:
        raise RuntimeError(f"Oracle target {target!r} is not visible in segmentation")
    return Detection(
        (float(columns.min()), float(rows.min()), float(columns.max()), float(rows.max())),
        world.objects[target].label,
        1.0,
    )


def resolve_scene_target(world: SimulationWorld, text_target: str) -> str:
    normalized = text_target.lower().strip()
    exact = [name for name, item in world.objects.items() if normalized in {name, item.label}]
    if len(exact) == 1:
        return exact[0]
    suffix = [
        name
        for name, item in world.objects.items()
        if normalized.endswith(name)
        and (not any(color in normalized.split() for color in ("red", "blue", "green", "yellow"))
             or item.color_name in normalized.split())
    ]
    if len(suffix) == 1:
        return suffix[0]
    raise ValueError(f"Target {text_target!r} does not uniquely match scene objects: {sorted(world.objects)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detection_prompts(config: dict[str, Any], target: str) -> list[str]:
    """Return configured open-vocabulary prompt ensemble without simulator truth."""
    configured = config.get("detection", {}).get("text_prompts", {})
    prompts = configured.get(target.lower().strip(), [target])
    result = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
    if not result:
        raise ValueError(f"No detection text prompts configured for target {target!r}")
    return list(dict.fromkeys(result))


def detection_image_size(config: dict[str, Any], target: str) -> int:
    detection = config.get("detection", {})
    configured = detection.get("image_size_by_target", {})
    return int(configured.get(target.lower().strip(), detection.get("image_size", 640)))


def _organized_world_points(observation: Any) -> tuple[np.ndarray, np.ndarray]:
    """Backproject every pixel without losing its image-grid correspondence."""
    height, width = observation.depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.column_stack((u.ravel(), v.ravel()))
    camera_points = backproject_pixels(pixels, observation.depth_m.ravel(), observation.intrinsic)
    world_points = transform_points(observation.T_world_camera, camera_points).reshape(height, width, 3)
    valid = np.isfinite(observation.depth_m) & (observation.depth_m > 0)
    return world_points, valid


def _graspnet_workspace_mask(
    observation: Any, config: dict[str, Any], target: str | None = None
) -> np.ndarray:
    world_points, valid = _organized_world_points(observation)
    minimum = np.asarray(config["pointcloud"]["workspace_min"], dtype=np.float64)
    maximum = np.asarray(config["pointcloud"]["workspace_max"], dtype=np.float64)
    mask = valid & np.all((world_points >= minimum) & (world_points <= maximum), axis=2)
    margin = float(config.get("graspnet", {}).get("exclude_table_plane_margin_m", 0.0))
    configured_targets = {
        str(item).strip().lower()
        for item in config.get("graspnet", {}).get("exclude_table_plane_targets", [])
    }
    should_exclude_table = not configured_targets or (
        target is not None and target.strip().lower() in configured_targets
    )
    if margin > 0.0 and should_exclude_table:
        mask &= world_points[:, :, 2] > float(config["scene"]["table_top_z"]) + margin
    return mask


def _semantic_target_cluster(
    observation: Any,
    detection: Detection,
    table_z_m: float,
    depth_tolerance_m: float,
) -> np.ndarray:
    """Build a soft target cloud from the detected box and its median depth."""
    world_points, valid = _organized_world_points(observation)
    height, width = observation.depth_m.shape
    x1, y1, x2, y2 = detection.bbox_xyxy
    xa, xb = max(0, int(np.floor(x1))), min(width, int(np.ceil(x2)) + 1)
    ya, yb = max(0, int(np.floor(y1))), min(height, int(np.ceil(y2)) + 1)
    box_valid = valid[ya:yb, xa:xb]
    box_depth = observation.depth_m[ya:yb, xa:xb]
    samples = box_depth[box_valid]
    if samples.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    reference = float(np.median(samples))
    depth_mask = np.abs(box_depth - reference) <= depth_tolerance_m
    points = world_points[ya:yb, xa:xb][box_valid & depth_mask]
    return points[points[:, 2] > table_z_m + 0.005]


def detect_scene(config: dict[str, Any], target: str, seed: int) -> tuple[Path, dict[str, Any]]:
    output = make_run_dir(config, "detect", seed)
    with SimulationWorld.create(config, seed) as world:
        target_key = resolve_scene_target(world, target)
        observation = world.camera.capture()
        truth = oracle_detection(world, observation, target_key)
        detection_config = config["detection"]
        configured_model = Path(str(detection_config["model"]))
        if not configured_model.is_absolute():
            configured_model = Path(config["_config_path"]).parent.parent / configured_model
        requested_device = str(detection_config.get("device", "auto"))
        if requested_device == "auto":
            import torch

            requested_device = "0" if torch.cuda.is_available() else "cpu"
        detector = YOLOWorldDetector(
            str(configured_model),
            confidence=float(detection_config["confidence_threshold"]),
            iou=float(detection_config["nms_iou_threshold"]),
            device=requested_device,
            image_size=detection_image_size(config, target),
            max_detections=int(detection_config.get("max_detections", 50)),
            fallback_confidence=float(detection_config["fallback_confidence_threshold"])
            if detection_config.get("fallback_confidence_threshold") is not None else None,
            fallback_min_prompt_votes=int(detection_config.get("fallback_min_prompt_votes", 2)),
            fallback_consensus_iou=float(detection_config.get("fallback_consensus_iou", 0.55)),
            fallback_maximum_area_fraction=float(
                detection_config.get("fallback_maximum_box_area_fraction", 0.08)
            ),
            fallback_reject_border_touching=bool(
                detection_config.get("fallback_reject_border_touching", True)
            ),
            force_prompt_consensus=target.lower().strip() in {
                str(value).lower().strip()
                for value in detection_config.get("force_prompt_consensus_targets", [])
            },
        )
        prompts = detection_prompts(config, target)
        detections = detector.detect(observation.rgb, prompts)
        metrics = detection_metrics(
            detections, truth, float(detection_config.get("metric_iou_threshold", 0.25))
        )
        metrics.update(
            {
                "mode": "real_yolo_world",
                "prompt": target,
                "text_prompts": prompts,
                "resolved_truth_object": target_key,
                "device": requested_device,
                "wall_inference_s": detector.last_inference_s,
                "ultralytics_speed_ms": detector.last_speed_ms,
                "fallback_used": detector.last_retry_used,
                "primary_prediction_count": detector.last_primary_count,
                "fallback_raw_prediction_count": detector.last_fallback_raw_count,
                "consensus_votes": detector.last_consensus_votes,
                "fallback_geometry_rejected": detector.last_fallback_geometry_rejected,
                "model_path": str(detector.model_path),
                "model_sha256": sha256_file(detector.model_path),
            }
        )
        save_rgb(output / "rgb.png", observation.rgb)
        save_detections(output / "raw_predictions.json", detections)
        save_detection_overlay(output / "detections.png", observation.rgb, detections)
        save_detections(output / "oracle_truth.json", [truth])
        save_detection_overlay(output / "oracle_truth.png", observation.rgb, [truth])
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output, metrics


def capture_artifacts(config: dict[str, Any], seed: int) -> tuple[Path, dict[str, Any]]:
    output = make_run_dir(config, "capture", seed)
    with SimulationWorld.create(config, seed) as world:
        observation = world.camera.capture()
        points, colors = rgbd_to_pointcloud(
            observation.rgb, observation.depth_m, observation.intrinsic, observation.T_world_camera
        )
        points, colors = crop_workspace(
            points,
            colors,
            np.asarray(config["pointcloud"]["workspace_min"]),
            np.asarray(config["pointcloud"]["workspace_max"]),
        )
        save_rgb(output / "rgb.png", observation.rgb)
        depth_preview = np.clip(observation.depth_m / observation.far_m * 65535, 0, 65535).astype(np.uint16)
        iio.imwrite(output / "depth_mm.png", np.clip(observation.depth_m * 1000, 0, 65535).astype(np.uint16))
        iio.imwrite(output / "depth_preview.png", depth_preview)
        iio.imwrite(output / "segmentation.png", segmentation_body_ids(observation.segmentation).astype(np.uint16))
        write_ply(output / "pointcloud.ply", points, colors)
        save_topdown_preview(output / "pointcloud_topdown.png", points, colors)
        np.savez_compressed(
            output / "camera.npz",
            intrinsic=observation.intrinsic,
            T_world_camera=observation.T_world_camera,
            depth_m=observation.depth_m,
        )
        metrics: dict[str, Any] = {"points": len(points), "objects": {}}
        body_ids = segmentation_body_ids(observation.segmentation)
        all_points, _ = rgbd_to_pointcloud(
            observation.rgb, observation.depth_m, observation.intrinsic, observation.T_world_camera
        )
        for name, scene_object in world.objects.items():
            mask = body_ids.ravel() == scene_object.body_id
            estimated = np.median(all_points[mask], axis=0) if np.any(mask) else np.full(3, np.nan)
            aabb_minimum, aabb_maximum = world.object_aabb(name)
            truth = (aabb_minimum + aabb_maximum) / 2.0
            metrics["objects"][name] = {
                "truth_world_m": truth.tolist(),
                "visible_surface_median_world_m": estimated.tolist(),
                "surface_to_center_error_m": float(np.linalg.norm(estimated - truth)),
            }
        metrics["table_surface_median_z_m"] = float(
            np.median(all_points[body_ids.ravel() == world.table_id, 2])
        )
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output, metrics


def export_graspnet_request(config: dict[str, Any], seed: int) -> tuple[Path, dict[str, Any]]:
    """Capture PyBullet RGB-D and validate the isolated-service request boundary."""
    output = make_run_dir(config, "graspnet_request", seed)
    with SimulationWorld.create(config, seed) as world:
        observation = world.camera.capture()
        workspace_mask = np.isfinite(observation.depth_m) & (observation.depth_m > 0)
        request = output / "request.npz"
        GraspNetFileClient.write_request(
            request, observation.rgb, observation.depth_m, observation.intrinsic, workspace_mask
        )
        checked = GraspNetFileClient.read_request(request)
        points, colors = rgbd_to_pointcloud(
            checked["rgb"], checked["depth_m"], checked["intrinsic"]
        )
        save_rgb(output / "rgb.png", observation.rgb)
        write_ply(output / "pointcloud_camera.ply", points, colors)
    metrics = {
        "schema_version": "1.0",
        "valid_depth_points": int(len(points)),
        "request": str(request),
        "frame": "opencv_camera",
        "generator": "none_request_only",
    }
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output, metrics


def associate_scene(config: dict[str, Any], target: str, seed: int) -> tuple[Path, dict[str, Any]]:
    """Real YOLO-World plus scene-level geometric proposals; no IDs enter selection."""
    output = make_run_dir(config, "associate", seed)
    started = perf_counter()
    with SimulationWorld.create(config, seed) as world:
        target_key = resolve_scene_target(world, target)
        observation = world.camera.capture()
        truth = oracle_detection(world, observation, target_key)
        detection_config = config["detection"]
        model = Path(str(detection_config["model"]))
        if not model.is_absolute():
            model = Path(config["_config_path"]).parent.parent / model
        device = str(detection_config.get("device", "auto"))
        if device == "auto":
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        detector = YOLOWorldDetector(
            str(model),
            confidence=float(detection_config["confidence_threshold"]),
            iou=float(detection_config["nms_iou_threshold"]),
            device=device,
            image_size=detection_image_size(config, target),
            max_detections=int(detection_config.get("max_detections", 50)),
            fallback_confidence=float(detection_config["fallback_confidence_threshold"])
            if detection_config.get("fallback_confidence_threshold") is not None else None,
            fallback_min_prompt_votes=int(detection_config.get("fallback_min_prompt_votes", 2)),
            fallback_consensus_iou=float(detection_config.get("fallback_consensus_iou", 0.55)),
            fallback_maximum_area_fraction=float(
                detection_config.get("fallback_maximum_box_area_fraction", 0.08)
            ),
            fallback_reject_border_touching=bool(
                detection_config.get("fallback_reject_border_touching", True)
            ),
            force_prompt_consensus=target.lower().strip() in {
                str(value).lower().strip()
                for value in detection_config.get("force_prompt_consensus_targets", [])
            },
        )
        detections = detector.detect(observation.rgb, detection_prompts(config, target))
        all_points, all_colors = rgbd_to_pointcloud(
            observation.rgb, observation.depth_m, observation.intrinsic, observation.T_world_camera
        )
        points, colors = crop_workspace(
            all_points,
            all_colors,
            np.asarray(config["pointcloud"]["workspace_min"]),
            np.asarray(config["pointcloud"]["workspace_max"]),
        )
        association_config = config.get("association", {})
        candidates, clusters = generate_scene_clusters(
            points,
            table_z_m=float(config["scene"]["table_top_z"]),
            eps_m=float(association_config.get("cluster_eps_m", 0.018)),
            min_points=int(association_config.get("cluster_min_points", 40)),
        )
        selected_detection = max(detections, key=lambda item: item.confidence) if detections else None
        records = []
        if selected_detection is not None:
            T_camera_world = invert_transform(observation.T_world_camera)
            candidates, records = associate_candidates_with_records(
                candidates,
                selected_detection,
                observation.depth_m,
                observation.intrinsic,
                T_camera_world,
                depth_tolerance_m=float(association_config.get("depth_tolerance_m", 0.07)),
                box_shrink_fraction=float(association_config.get("box_shrink_fraction", 0.10)),
            )
        else:
            for candidate in candidates:
                candidate.accepted = False
                candidate.rejection_reasons.append("no_target_detection")
        ranked = rank_candidates(candidates, config["scoring"])

        # Simulation truth is consulted only after selection, solely for metrics.
        target_min, target_max = world.object_aabb(target_key)
        target_center = (target_min + target_max) / 2.0
        accepted_candidates = [candidate for candidate in ranked if candidate.accepted]
        selected_candidate = accepted_candidates[0] if accepted_candidates else None
        selected_target_error = (
            float(np.linalg.norm(selected_candidate.center - target_center))
            if selected_candidate is not None else None
        )
        metrics = {
            "mode": "real_yolo_world + geometric_scene_baseline",
            "target": target,
            "resolved_truth_object": target_key,
            "detection_count": len(detections),
            "cluster_count": len(clusters),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted_candidates),
            "association_success": bool(accepted_candidates),
            "selected_candidate_target_center_error_m": selected_target_error,
            "selected_candidate_matches_target_truth": bool(
                selected_target_error is not None
                and selected_target_error <= float(association_config.get("truth_center_tolerance_m", 0.08))
            ),
            "detector_wall_s": detector.last_inference_s,
            "elapsed_s": perf_counter() - started,
            "truth_used_for_selection": False,
            "candidate_generator": "geometric_scene_baseline_not_graspnet",
        }
        save_rgb(output / "rgb.png", observation.rgb)
        save_detections(output / "raw_predictions.json", detections)
        save_detection_overlay(output / "detections.png", observation.rgb, detections)
        save_detections(output / "oracle_truth.json", [truth])
        if selected_detection is not None:
            save_association_overlay(output / "association_2d.png", observation.rgb, selected_detection, records)
        write_ply(output / "scene_pointcloud.ply", points, colors)
        centers = np.asarray([candidate.center for candidate in candidates], dtype=np.float64).reshape(-1, 3)
        kept = np.asarray([candidate.accepted for candidate in candidates], dtype=bool)
        save_candidate_topdown(output / "association_3d_topdown.png", points, colors, centers, kept)
        (output / "association_records.json").write_text(
            json.dumps([record.to_dict() for record in records], indent=2), encoding="utf-8"
        )
        (output / "candidates.json").write_text(
            json.dumps([candidate.to_dict() for candidate in ranked], indent=2), encoding="utf-8"
        )
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output, metrics


def run_open_vocab_grasp(
    config: dict[str, Any], target: str | None, seed: int, *, semantic_selection: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Execute real YOLO selection with geometric or isolated official GraspNet proposals."""
    target = target or str(config["grasp"]["target"])
    output = make_run_dir(config, "run", seed)
    started = perf_counter()
    with SimulationWorld.create(config, seed) as world:
        target_key = resolve_scene_target(world, target)
        observation = world.camera.capture()
        truth = oracle_detection(world, observation, target_key)
        detection_config = config["detection"]
        detector: YOLOWorldDetector | None = None
        if semantic_selection:
            model = Path(str(detection_config["model"]))
            if not model.is_absolute():
                model = Path(config["_config_path"]).parent.parent / model
            device = str(detection_config.get("device", "auto"))
            if device == "auto":
                import torch

                device = "0" if torch.cuda.is_available() else "cpu"
            detector = YOLOWorldDetector(
                str(model),
                confidence=float(detection_config["confidence_threshold"]),
                iou=float(detection_config["nms_iou_threshold"]),
                device=device,
                image_size=detection_image_size(config, target),
                max_detections=int(detection_config.get("max_detections", 50)),
                fallback_confidence=float(detection_config["fallback_confidence_threshold"])
                if detection_config.get("fallback_confidence_threshold") is not None else None,
                fallback_min_prompt_votes=int(detection_config.get("fallback_min_prompt_votes", 2)),
                fallback_consensus_iou=float(detection_config.get("fallback_consensus_iou", 0.55)),
                fallback_maximum_area_fraction=float(
                    detection_config.get("fallback_maximum_box_area_fraction", 0.08)
                ),
                fallback_reject_border_touching=bool(
                    detection_config.get("fallback_reject_border_touching", True)
                ),
                force_prompt_consensus=target.lower().strip() in {
                    str(value).lower().strip()
                    for value in detection_config.get("force_prompt_consensus_targets", [])
                },
            )
            detections = detector.detect(observation.rgb, detection_prompts(config, target))
            detector_metrics = detection_metrics(
                detections, truth, float(detection_config.get("metric_iou_threshold", 0.25))
            )
        else:
            detections = []
            detector_metrics = {
                "detection_success": None,
                "target_selection_correct": None,
                "selected_iou": None,
            }
        all_points, all_colors = rgbd_to_pointcloud(
            observation.rgb, observation.depth_m, observation.intrinsic, observation.T_world_camera
        )
        points, colors = crop_workspace(
            all_points,
            all_colors,
            np.asarray(config["pointcloud"]["workspace_min"]),
            np.asarray(config["pointcloud"]["workspace_max"]),
        )
        association_config = config.get("association", {})
        selected_detection = max(detections, key=lambda item: item.confidence) if detections else None
        generator_name = str(config["grasp"].get("generator", "geometric_baseline"))
        if not semantic_selection and generator_name != "graspnet":
            raise ValueError("graspnet-only mode requires grasp.generator: graspnet")
        generation_started = perf_counter()
        graspnet_metadata = None
        if generator_name == "geometric_baseline":
            candidates, clusters = generate_scene_clusters(
                points,
                table_z_m=float(config["scene"]["table_top_z"]),
                eps_m=float(association_config.get("cluster_eps_m", 0.018)),
                min_points=int(association_config.get("cluster_min_points", 40)),
            )
            candidate_generator = "geometric_scene_baseline_not_graspnet"
            result_mode = "open-vocab-simple"
        elif generator_name == "graspnet":
            project_root = Path(config["_config_path"]).parent.parent
            request_path = output / "graspnet_request.npz"
            response_path = output / "graspnet_response.npz"
            GraspNetFileClient.write_request(
                request_path,
                observation.rgb,
                observation.depth_m,
                observation.intrinsic,
                _graspnet_workspace_mask(observation, config, target_key),
            )
            candidates, graspnet_metadata = GraspNetFileClient.run_isolated_inference(
                request_path, response_path, project_root, config
            )
            candidates = camera_grasps_to_world_tool(candidates, observation.T_world_camera)
            clusters = []
            candidate_generator = "official_graspnet_checkpoint_rs"
            result_mode = "open-vocab-graspnet" if semantic_selection else "graspnet-only"
            (output / "graspnet_stdout.log").write_text(graspnet_metadata.stdout, encoding="utf-8")
            (output / "graspnet_stderr.log").write_text(graspnet_metadata.stderr, encoding="utf-8")
        else:
            raise ValueError(
                f"Unsupported grasp.generator {generator_name!r}; expected geometric_baseline or graspnet"
            )
        generation_s = perf_counter() - generation_started
        records = []
        semantic_target_center: np.ndarray | None = None
        if not semantic_selection:
            scene_proposals, clusters = generate_scene_clusters(
                points,
                table_z_m=float(config["scene"]["table_top_z"]),
                eps_m=float(association_config.get("cluster_eps_m", 0.018)),
                min_points=int(association_config.get("cluster_min_points", 40)),
            )
            cluster_centers: dict[int, np.ndarray] = {}
            for proposal in scene_proposals:
                if proposal.source_cluster_id is not None:
                    cluster_centers.setdefault(proposal.source_cluster_id, proposal.center.copy())
            if cluster_centers:
                ordered_ids = sorted(cluster_centers)
                centers = np.asarray([cluster_centers[index] for index in ordered_ids])
                for candidate in candidates:
                    nearest = int(np.argmin(np.linalg.norm(centers - candidate.center, axis=1)))
                    cluster_id = ordered_ids[nearest]
                    candidate.source_cluster_id = cluster_id
                    delta = cluster_centers[cluster_id] - candidate.center
                    candidate.semantic_center_distance_m = float(np.linalg.norm(delta))
                    candidate.height_axis_offset_m = float(
                        abs(np.dot(candidate.rotation[:, 0], delta))
                    )
        elif selected_detection is not None:
            candidates, records = associate_candidates_with_records(
                candidates,
                selected_detection,
                observation.depth_m,
                observation.intrinsic,
                invert_transform(observation.T_world_camera),
                depth_tolerance_m=float(association_config.get("depth_tolerance_m", 0.07)),
                box_shrink_fraction=float(association_config.get("box_shrink_fraction", 0.10)),
            )
            if generator_name == "graspnet":
                semantic_proposals, semantic_clusters = generate_scene_clusters(
                    points,
                    table_z_m=float(config["scene"]["table_top_z"]),
                    eps_m=float(association_config.get("cluster_eps_m", 0.018)),
                    min_points=int(association_config.get("cluster_min_points", 40)),
                )
                semantic_proposals, _ = associate_candidates_with_records(
                    semantic_proposals,
                    selected_detection,
                    observation.depth_m,
                    observation.intrinsic,
                    invert_transform(observation.T_world_camera),
                    depth_tolerance_m=float(association_config.get("depth_tolerance_m", 0.07)),
                    box_shrink_fraction=float(association_config.get("box_shrink_fraction", 0.10)),
                )
                semantic_matches = [proposal for proposal in semantic_proposals if proposal.accepted]
                if semantic_matches:
                    reference = max(semantic_matches, key=lambda proposal: proposal.center_score)
                    semantic_target_center = reference.center.copy()
                    target_cluster = semantic_clusters[reference.source_cluster_id or 0]
                    clusters = [target_cluster]
                    for candidate in candidates:
                        if candidate.accepted:
                            candidate.source_cluster_id = 0
                            delta = semantic_target_center - candidate.center
                            candidate.semantic_center_distance_m = float(np.linalg.norm(delta))
                            candidate.height_axis_offset_m = float(
                                abs(np.dot(candidate.rotation[:, 0], delta))
                            )
                else:
                    target_cluster = _semantic_target_cluster(
                        observation,
                        selected_detection,
                        float(config["scene"]["table_top_z"]),
                        float(association_config.get("depth_tolerance_m", 0.07)),
                    )
                    clusters = [target_cluster] if len(target_cluster) else []
                    if clusters:
                        semantic_target_center = np.median(target_cluster, axis=0)
                        for candidate in candidates:
                            if candidate.accepted:
                                candidate.source_cluster_id = 0
                                delta = semantic_target_center - candidate.center
                                candidate.semantic_center_distance_m = float(np.linalg.norm(delta))
                                candidate.height_axis_offset_m = float(
                                    abs(np.dot(candidate.rotation[:, 0], delta))
                                )
        else:
            for candidate in candidates:
                candidate.accepted = False
                candidate.rejection_reasons.append("no_target_detection")

        planning_started = perf_counter()
        filter_summary = filter_grasp_candidates(
            world,
            candidates,
            points,
            clusters,
            world.objects[target_key].body_id if semantic_selection else None,
            config,
            semantic_target_center_world=semantic_target_center,
            allowed_contact_body_ids=(
                {item.body_id for item in world.objects.values()}
                if not semantic_selection else None
            ),
        )
        ranked = rank_candidates(candidates, config["scoring"])
        planning_s = perf_counter() - planning_started
        selected = next((candidate for candidate in ranked if candidate.accepted), None)
        # Truth is sampled only after perception/ranking and before execution so
        # the metric is not contaminated by the subsequent object lift.
        target_min, target_max = world.object_aabb(target_key)
        truth_center_before_execution = (target_min + target_max) / 2.0
        selected_truth_error = (
            float(np.linalg.norm(selected.center - truth_center_before_execution))
            if selected is not None else None
        )
        final_records = [
            replace(
                record,
                accepted=candidates[record.candidate_index].accepted,
                rejection_reasons=tuple(candidates[record.candidate_index].rejection_reasons),
            )
            for record in records
        ]
        video_frames: list[np.ndarray] = [observation.rgb]

        def capture_video_frame() -> None:
            video_frames.append(world.camera.capture().rgb)

        execution_started = perf_counter()
        if selected is None:
            if semantic_selection and not detections:
                terminal_reason = "detection_failed"
            elif not candidates:
                terminal_reason = "candidate_generation_failed"
            elif filter_summary.stage_counts["associated"] == 0:
                terminal_reason = "association_failed"
            else:
                terminal_reason = "no_accepted_candidate"
            result: dict[str, Any] = {
                "success": False,
                "failure_reason": terminal_reason,
                "state_history": ["RESET", "OBSERVE", "DETECT", "GENERATE_GRASPS", "SELECT_GRASP", "FAILED"],
                "metrics": {},
            }
        else:
            if bool(config["simulation"].get("gui", False)):
                world.add_grasp_axes(selected.center, selected.rotation)
            execution = GraspExecutor(world, capture_video_frame).execute(selected, target_key)
            result = {
                "success": execution.success,
                "failure_reason": execution.failure_reason,
                "state_history": execution.state_history,
                "metrics": execution.metrics,
            }
        execution_s = perf_counter() - execution_started

        result.update(
            {
                "mode": result_mode,
                "perception": "real_yolo_world" if semantic_selection else "none_graspnet_only",
                "candidate_generator": candidate_generator,
                "target": target,
                "resolved_truth_object": target_key,
                "seed": seed,
                "detection_count": len(detections),
                "detection_success": detector_metrics["detection_success"],
                "target_selection_correct": detector_metrics["target_selection_correct"],
                "selected_detection_iou": detector_metrics["selected_iou"],
                "cluster_count": len(clusters),
                "candidate_count": len(candidates),
                "candidate_generation_success": bool(candidates),
                "accepted_count": sum(candidate.accepted for candidate in candidates),
                "ik_reachable": filter_summary.stage_counts["ik_and_trajectory"] > 0,
                "filter_stage_counts": filter_summary.stage_counts,
                "rejection_counts": filter_summary.rejection_counts,
                "selected_candidate_target_center_error_m": selected_truth_error,
                "truth_used_for_semantic_selection": False,
                "simulation_body_ids_used_for_physics_collision_only": True,
                "detector_wall_s": detector.last_inference_s if detector is not None else 0.0,
                "detection_fallback_used": detector.last_retry_used if detector is not None else False,
                "detection_primary_count": detector.last_primary_count if detector is not None else 0,
                "detection_fallback_raw_count": (
                    detector.last_fallback_raw_count if detector is not None else 0
                ),
                "detection_consensus_votes": (
                    detector.last_consensus_votes if detector is not None else []
                ),
                "detection_fallback_geometry_rejected": (
                    detector.last_fallback_geometry_rejected if detector is not None else 0
                ),
                "grasp_generation_s": generation_s,
                "graspnet_inference_s": (
                    graspnet_metadata.wall_s if graspnet_metadata is not None else None
                ),
                "graspnet_raw_collision_free_count": (
                    sum(not candidate.collision for candidate in candidates)
                    if generator_name == "graspnet" else None
                ),
                "semantic_target_center_world_m": (
                    semantic_target_center.tolist() if semantic_target_center is not None else None
                ),
                "semantic_geometry_source": (
                    (
                        "rgbd_scene_clusters_without_text_or_simulation_truth"
                        if not semantic_selection
                        else "rgbd_scene_cluster_not_simulation_truth"
                    )
                    if generator_name == "graspnet" else None
                ),
                "planning_s": planning_s,
                "execution_s": execution_s,
                "elapsed_s": perf_counter() - started,
            }
        )
        save_rgb(output / "rgb.png", observation.rgb)
        save_detections(output / "raw_predictions.json", detections)
        save_detection_overlay(output / "detections.png", observation.rgb, detections)
        save_detections(output / "oracle_truth.json", [truth])
        if selected_detection is not None:
            save_association_overlay(
                output / "filtered_candidates_2d.png", observation.rgb, selected_detection, final_records
            )
        write_ply(output / "scene_pointcloud.ply", points, colors)
        centers = np.asarray([candidate.center for candidate in candidates], dtype=np.float64).reshape(-1, 3)
        kept = np.asarray([candidate.accepted for candidate in candidates], dtype=bool)
        save_candidate_topdown(output / "filtered_candidates_3d.png", points, colors, centers, kept)
        (output / "association_records.json").write_text(
            json.dumps([record.to_dict() for record in final_records], indent=2), encoding="utf-8"
        )
        (output / "candidates.json").write_text(
            json.dumps([candidate.to_dict() for candidate in ranked], indent=2), encoding="utf-8"
        )
        iio.imwrite(output / "demo.gif", np.stack(video_frames), duration=80, loop=0)
        iio.imwrite(output / "demo.mp4", np.stack(video_frames), fps=12)
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output, result


def run_graspnet_only(
    config: dict[str, Any], target: str | None, seed: int
) -> tuple[Path, dict[str, Any]]:
    """Run the semantic-free official-GraspNet baseline.

    ``target`` is used only for post-selection success evaluation. Candidate
    generation, clustering, filtering and ranking receive no text box or target
    body identity.
    """
    return run_open_vocab_grasp(config, target, seed, semantic_selection=False)


def run_cpu_smoke(config: dict[str, Any], target: str | None, seed: int) -> tuple[Path, dict[str, Any]]:
    target = target or str(config["grasp"]["target"])
    output = make_run_dir(config, "smoke", seed)
    started = perf_counter()
    with SimulationWorld.create(config, seed) as world:
        if target not in world.objects:
            raise ValueError(f"Target {target!r} not in scene: {sorted(world.objects)}")
        observation = world.camera.capture()
        detection = oracle_detection(world, observation, target)
        all_points, all_colors = rgbd_to_pointcloud(
            observation.rgb, observation.depth_m, observation.intrinsic, observation.T_world_camera
        )
        scene_points, _ = crop_workspace(
            all_points,
            all_colors,
            np.asarray(config["pointcloud"]["workspace_min"]),
            np.asarray(config["pointcloud"]["workspace_max"]),
        )
        body_ids = segmentation_body_ids(observation.segmentation)
        target_points = all_points[body_ids.ravel() == world.objects[target].body_id]
        generation_started = perf_counter()
        candidates = GeometricTopDownGenerator().generate_from_points(target_points)
        generation_s = perf_counter() - generation_started
        planning_started = perf_counter()
        for candidate in candidates:
            candidate.detection_confidence = 1.0
            candidate.center_score = 1.0
            candidate.source_cluster_id = 0
        filter_summary = filter_grasp_candidates(
            world,
            candidates,
            scene_points,
            [target_points],
            world.objects[target].body_id,
            config,
        )
        ranked = rank_candidates(candidates, config["scoring"])
        planning_s = perf_counter() - planning_started
        selected = next((candidate for candidate in ranked if candidate.accepted), None)
        video_frames: list[np.ndarray] = [observation.rgb]
        def capture_video_frame() -> None:
            video_frames.append(world.camera.capture().rgb)
        execution_started = perf_counter()
        if selected is None:
            result = {"success": False, "failure_reason": "no_accepted_candidate", "state_history": []}
        else:
            if bool(config["simulation"].get("gui", False)):
                world.add_grasp_axes(selected.center, selected.rotation)
            execution = GraspExecutor(world, capture_video_frame).execute(selected, target)
            result = {
                "success": execution.success,
                "failure_reason": execution.failure_reason,
                "state_history": execution.state_history,
                "metrics": execution.metrics,
            }
        execution_s = perf_counter() - execution_started
        result.update(
            {
                "mode": "oracle-perception + geometric_baseline",
                "target": target,
                "seed": seed,
                "candidate_count": len(candidates),
                "accepted_count": sum(candidate.accepted for candidate in candidates),
                "filter_stage_counts": filter_summary.stage_counts,
                "rejection_counts": filter_summary.rejection_counts,
                "detector_wall_s": 0.0,
                "grasp_generation_s": generation_s,
                "planning_s": planning_s,
                "execution_s": execution_s,
                "elapsed_s": perf_counter() - started,
            }
        )
        save_rgb(output / "rgb.png", observation.rgb)
        save_detection_overlay(output / "detection_oracle.png", observation.rgb, [detection])
        write_ply(output / "pointcloud.ply", all_points, all_colors)
        iio.imwrite(output / "demo.gif", np.stack(video_frames), duration=80, loop=0)
        (output / "candidates.json").write_text(
            json.dumps([candidate.to_dict() for candidate in ranked], indent=2), encoding="utf-8"
        )
    save_config(config, output / "config.yaml")
    write_environment_snapshot(output / "environment.json")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output, result
