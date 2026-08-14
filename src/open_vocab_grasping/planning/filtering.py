from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from open_vocab_grasping.grasping.interface import GraspCandidate
from open_vocab_grasping.grasping.contact import (
    closing_axis_center_shift,
    contact_support_decision,
    parallel_jaw_symmetric_rotation,
    parallel_jaw_contact_quality,
    tabletop_topdown_rotation,
)
from open_vocab_grasping.planning.collision import (
    joint_path_collision_free,
    pointcloud_clearance,
    table_clearance_ok,
)
from open_vocab_grasping.planning.ik import solve_ik
from open_vocab_grasping.planning.trajectory import interpolate_joints
from open_vocab_grasping.simulation.world import SimulationWorld


@dataclass(frozen=True)
class FilterSummary:
    stage_counts: dict[str, int]
    rejection_counts: dict[str, int]


def approach_distance_options(nominal_m: float, minimum_m: float, step_m: float) -> list[float]:
    if nominal_m <= 0 or minimum_m <= 0 or step_m <= 0:
        raise ValueError("approach distances and step must be positive")
    if minimum_m > nominal_m:
        raise ValueError("minimum approach distance cannot exceed nominal distance")
    count = int(np.floor((nominal_m - minimum_m) / step_m + 1e-9))
    values = [nominal_m - index * step_m for index in range(count + 1)]
    if values[-1] > minimum_m + 1e-9:
        values.append(minimum_m)
    return [float(value) for value in values]


def _reject(candidate: GraspCandidate, reason: str) -> None:
    candidate.accepted = False
    if reason not in candidate.rejection_reasons:
        candidate.rejection_reasons.append(reason)


def filter_grasp_candidates(
    world: SimulationWorld,
    candidates: list[GraspCandidate],
    scene_points_world: np.ndarray,
    clusters: list[np.ndarray],
    target_body_id: int | None,
    config: dict[str, Any],
    semantic_target_center_world: np.ndarray | None = None,
    allowed_contact_body_ids: set[int] | None = None,
) -> FilterSummary:
    """Apply deterministic physical and kinematic filters to associated candidates."""
    filter_config = config["filters"]
    grasp_config = config["grasp"]
    workspace_min = np.asarray(config["pointcloud"]["workspace_min"], dtype=np.float64)
    workspace_max = np.asarray(config["pointcloud"]["workspace_max"], dtype=np.float64)
    obstacles = [world.table_id, *(item.body_id for item in world.objects.values())]
    allowed_contacts = set(allowed_contact_body_ids or set())
    if target_body_id is not None:
        allowed_contacts.add(target_body_id)
    stage_counts: dict[str, int] = {
        "generated": len(candidates),
        "associated": sum(candidate.accepted for candidate in candidates),
    }

    contact_parameters = {
        "gripper_height_m": float(filter_config.get("contact_gripper_height_m", 0.03)),
        "contact_depth_m": float(filter_config.get("contact_depth_m", 0.05)),
        "minimum_side_offset_m": float(
            filter_config.get("contact_minimum_side_offset_m", 0.003)
        ),
        "minimum_points": int(filter_config.get("contact_minimum_points", 12)),
    }

    if bool(filter_config.get("enable_topdown_orientation_refinement", False)):
        minimum_alignment = float(filter_config["minimum_topdown_alignment"])
        for candidate in candidates:
            if not candidate.accepted or candidate.collision:
                continue
            approach = np.asarray(candidate.rotation[:, 2], dtype=np.float64)
            alignment = float(np.dot(approach / np.linalg.norm(approach), [0.0, 0.0, -1.0]))
            if alignment >= minimum_alignment:
                continue
            refined_rotation = tabletop_topdown_rotation(candidate.rotation)
            if refined_rotation is None:
                continue
            candidate.rotation = refined_rotation
            candidate.used_topdown_orientation_refinement = True
            if semantic_target_center_world is not None:
                delta = np.asarray(semantic_target_center_world) - candidate.center
                candidate.semantic_center_distance_m = float(np.linalg.norm(delta))
                candidate.height_axis_offset_m = float(
                    abs(np.dot(candidate.rotation[:, 0], delta))
                )

    # GraspNet predicts the orientation and contact depth. In synthetic RGB-D,
    # its sampled center can land on a visible mug wall. Optionally translate
    # only within the gripper plane toward the semantic RGB-D center. The move
    # is retained only when visible bilateral support improves; network rotation,
    # depth, score and collision flag remain untouched and auditable.
    refinement_limit = float(filter_config.get("semantic_plane_refinement_max_m", 0.0))
    refinement_trigger = float(filter_config.get("semantic_plane_refinement_trigger_m", 0.02))
    if semantic_target_center_world is not None and refinement_limit > 0.0:
        semantic_center = np.asarray(semantic_target_center_world, dtype=np.float64)
        for candidate in candidates:
            if not candidate.accepted or candidate.collision:
                continue
            cluster = (
                clusters[candidate.source_cluster_id]
                if candidate.source_cluster_id is not None and candidate.source_cluster_id < len(clusters)
                else np.empty((0, 3))
            )
            delta = semantic_center - candidate.center
            if np.linalg.norm(delta) <= refinement_trigger or not len(cluster):
                continue
            approach_axis = candidate.rotation[:, 2]
            planar_delta = delta - approach_axis * float(np.dot(delta, approach_axis))
            planar_norm = float(np.linalg.norm(planar_delta))
            if planar_norm <= 1e-9:
                continue
            planar_shift = planar_delta * min(1.0, refinement_limit / planar_norm)
            original_quality = parallel_jaw_contact_quality(
                candidate, cluster, **contact_parameters
            )
            original_center = candidate.center.copy()
            original_distance = float(np.linalg.norm(delta))
            candidate.center = original_center + planar_shift
            refined_quality = parallel_jaw_contact_quality(
                candidate, cluster, **contact_parameters
            )
            candidate.semantic_refinement_original_contact_score = original_quality.score
            candidate.semantic_refinement_contact_score = refined_quality.score
            refined_delta = semantic_center - candidate.center
            refined_distance = float(np.linalg.norm(refined_delta))
            if (
                refined_quality.score > original_quality.score
                or (
                    refined_quality.score
                    >= float(filter_config.get("minimum_contact_score", 0.0))
                    and refined_distance + 1e-6 < original_distance
                )
            ):
                candidate.semantic_plane_shift_m = float(np.linalg.norm(planar_shift))
                candidate.semantic_center_distance_m = refined_distance
                candidate.height_axis_offset_m = float(
                    abs(np.dot(candidate.rotation[:, 0], refined_delta))
                )
            else:
                candidate.center = original_center

    for candidate in candidates:
        if not candidate.accepted:
            continue
        if candidate.collision:
            _reject(candidate, "graspnet_model_free_collision")
        if candidate.grasp_score < float(filter_config["minimum_grasp_score"]):
            _reject(candidate, "grasp_score_below_threshold")
        maximum_predicted_width = float(
            filter_config.get("maximum_predicted_width_m", grasp_config["max_width_m"])
        )
        if not float(filter_config["minimum_width_m"]) <= candidate.width <= maximum_predicted_width:
            _reject(candidate, "width_out_of_range")
        if not np.all((candidate.center >= workspace_min) & (candidate.center <= workspace_max)):
            _reject(candidate, "outside_robot_workspace")
        if not table_clearance_ok(candidate.center, float(filter_config["minimum_center_z_m"])):
            _reject(candidate, "table_penetration")
        maximum_height_offset = filter_config.get("maximum_target_height_axis_offset_m")
        if (
            maximum_height_offset is not None
            and candidate.height_axis_offset_m is not None
            and candidate.height_axis_offset_m > float(maximum_height_offset)
        ):
            _reject(candidate, "target_outside_gripper_height")
        # Replacing a network-predicted approach direction with a tabletop
        # top-down approach is intentionally conservative: the pose is only
        # retained when its center remains close to the target center estimated
        # from the YOLO box and RGB-D cluster. This prevents high network scores
        # on a bowl rim/background edge from outranking a centered grasp.
        maximum_refined_center_distance = filter_config.get(
            "maximum_topdown_refined_center_distance_m"
        )
        if (
            candidate.used_topdown_orientation_refinement
            and maximum_refined_center_distance is not None
            and candidate.semantic_center_distance_m is not None
            and candidate.semantic_center_distance_m
            > float(maximum_refined_center_distance)
        ):
            _reject(candidate, "topdown_refinement_too_far_from_semantic_center")
        approach = np.asarray(candidate.rotation[:, 2], dtype=np.float64)
        downward_alignment = float(np.dot(approach / np.linalg.norm(approach), np.array([0.0, 0.0, -1.0])))
        if downward_alignment < float(filter_config["minimum_topdown_alignment"]):
            _reject(candidate, "approach_direction_rejected")
    stage_counts["geometry"] = sum(candidate.accepted for candidate in candidates)

    for candidate in candidates:
        if not candidate.accepted:
            continue
        cluster = (
            clusters[candidate.source_cluster_id]
            if candidate.source_cluster_id is not None and candidate.source_cluster_id < len(clusters)
            else np.empty((0, 3))
        )
        quality = parallel_jaw_contact_quality(
            candidate,
            cluster,
            **contact_parameters,
        )
        original_center = candidate.center.copy()
        shift = closing_axis_center_shift(
            candidate,
            cluster,
            gripper_height_m=contact_parameters["gripper_height_m"],
            contact_depth_m=contact_parameters["contact_depth_m"],
            maximum_shift_m=float(filter_config.get("contact_center_refinement_max_m", 0.0)),
        )
        if len(cluster):
            cluster_center = (
                np.asarray(semantic_target_center_world, dtype=np.float64)
                if semantic_target_center_world is not None
                else np.median(cluster, axis=0)
            )
            center_seeking_shift = float(
                np.dot(cluster_center - candidate.center, candidate.rotation[:, 1])
            )
            # A partial RGB-D surface can make the raw closing-axis median point
            # toward a mug handle or rim. Never refine away from the semantic
            # cluster center, and do not overshoot that center by more than 5 mm.
            if shift * center_seeking_shift <= 0.0:
                shift = float(
                    np.clip(
                        center_seeking_shift,
                        -float(filter_config.get("contact_center_refinement_max_m", 0.0)),
                        float(filter_config.get("contact_center_refinement_max_m", 0.0)),
                    )
                )
            else:
                shift = float(
                    np.sign(shift) * min(abs(shift), abs(center_seeking_shift))
                )
        if abs(shift) > 1e-9:
            candidate.center = original_center + candidate.rotation[:, 1] * shift
            refined_quality = parallel_jaw_contact_quality(
                candidate,
                cluster,
                **contact_parameters,
            )
            if refined_quality.score > quality.score:
                quality = refined_quality
                candidate.contact_center_shift_m = shift
                if len(cluster):
                    refined_delta = cluster_center - candidate.center
                    candidate.semantic_center_distance_m = float(np.linalg.norm(refined_delta))
                    candidate.height_axis_offset_m = float(
                        abs(np.dot(candidate.rotation[:, 0], refined_delta))
                    )
            else:
                candidate.center = original_center
        candidate.contact_score = quality.score
        candidate.contact_balance = quality.balance
        candidate.enclosure_score = quality.enclosure
        candidate.target_points_in_gripper = quality.point_count
        minimum_contact_score = float(filter_config.get("minimum_contact_score", 0.0))
        approach_axis = np.asarray(candidate.rotation[:, 2], dtype=np.float64)
        downward_alignment = float(
            np.dot(approach_axis / np.linalg.norm(approach_axis), np.array([0.0, 0.0, -1.0]))
        )
        contact_supported, evidence_mode = contact_support_decision(
            quality,
            minimum_score=minimum_contact_score,
            semantic_center_distance_m=candidate.semantic_center_distance_m,
            downward_alignment=downward_alignment,
            occlusion_fallback_max_center_distance_m=float(
                filter_config.get("contact_occlusion_fallback_max_center_distance_m", 0.0)
            ),
            occlusion_fallback_min_alignment=float(
                filter_config.get("contact_occlusion_fallback_min_alignment", 1.0)
            ),
            occlusion_fallback_min_points=int(
                filter_config.get("contact_occlusion_fallback_min_points", 10**9)
            ),
        )
        candidate.contact_evidence_mode = evidence_mode
        if not contact_supported:
            _reject(candidate, "insufficient_bilateral_contact_support")
    stage_counts["contact_support"] = sum(candidate.accepted for candidate in candidates)

    for candidate in candidates:
        if not candidate.accepted:
            continue
        cluster = (
            clusters[candidate.source_cluster_id]
            if candidate.source_cluster_id is not None and candidate.source_cluster_id < len(clusters)
            else np.empty((0, 3))
        )
        clear, clearance = pointcloud_clearance(
            candidate.center,
            scene_points_world,
            cluster,
            float(filter_config["target_exclusion_margin_m"]),
            float(filter_config["minimum_scene_clearance_m"]),
        )
        candidate.clearance = clearance if np.isfinite(clearance) else 1.0
        if not clear:
            _reject(candidate, "scene_pointcloud_collision")
    stage_counts["pointcloud_collision"] = sum(candidate.accepted for candidate in candidates)

    for candidate in candidates:
        if not candidate.accepted:
            continue
        approach_axis = candidate.rotation[:, 2]
        grasp_position = candidate.center - approach_axis * float(grasp_config["tool_offset_m"])
        lift_position = grasp_position + np.array(
            [0.0, 0.0, float(grasp_config["lift_distance_m"])], dtype=np.float64
        )
        nominal_approach = float(grasp_config["approach_distance_m"])
        approach_distances = approach_distance_options(
            nominal_approach,
            float(filter_config.get("minimum_approach_distance_m", nominal_approach)),
            float(filter_config.get("approach_distance_step_m", 0.01)),
        )
        rotations = [(candidate.rotation, False)]
        if bool(filter_config.get("try_parallel_jaw_symmetry", True)):
            rotations.append((parallel_jaw_symmetric_rotation(candidate.rotation), True))

        chosen = None
        first_attempt = None
        for approach_distance in approach_distances:
            pregrasp_position_attempt = grasp_position - approach_axis * approach_distance
            for rotation, symmetric in rotations:
                pre_attempt = solve_ik(world.robot, pregrasp_position_attempt, rotation)
                grasp_attempt = solve_ik(world.robot, grasp_position, rotation)
                lift_attempt = solve_ik(world.robot, lift_position, rotation)
                if first_attempt is None:
                    first_attempt = (
                        pregrasp_position_attempt, pre_attempt, grasp_attempt, lift_attempt
                    )
                if pre_attempt.reachable and grasp_attempt.reachable and lift_attempt.reachable:
                    chosen = (
                        approach_distance,
                        pregrasp_position_attempt,
                        rotation,
                        symmetric,
                        pre_attempt,
                        grasp_attempt,
                        lift_attempt,
                    )
                    break
            if chosen is not None:
                break
        if chosen is not None:
            (
                candidate.approach_distance_m,
                pregrasp_position,
                candidate.rotation,
                candidate.used_parallel_jaw_symmetry,
                pre_ik,
                grasp_ik,
                lift_ik,
            ) = chosen
        else:
            assert first_attempt is not None
            pregrasp_position, pre_ik, grasp_ik, lift_ik = first_attempt
            candidate.approach_distance_m = nominal_approach
        candidate.reachable = pre_ik.reachable and grasp_ik.reachable and lift_ik.reachable
        if not pre_ik.reachable:
            _reject(candidate, "pregrasp_ik_unreachable")
        if not grasp_ik.reachable:
            _reject(candidate, "grasp_ik_unreachable")
        if not lift_ik.reachable:
            _reject(candidate, "lift_ik_unreachable")
        if pre_ik.reachable:
            candidate.motion_cost = float(np.linalg.norm(pre_ik.joints - world.robot.home))
        else:
            candidate.motion_cost = float(filter_config["unreachable_motion_cost"])
        if not candidate.accepted:
            continue
        home_to_pre = interpolate_joints(
            world.robot.home, pre_ik.joints, int(filter_config["collision_check_steps"])
        )
        collision_free, reason = joint_path_collision_free(world.robot, home_to_pre, obstacles)
        if not collision_free:
            _reject(candidate, f"home_to_pregrasp_{reason}")
            continue
        pre_to_grasp = interpolate_joints(
            pre_ik.joints, grasp_ik.joints, int(filter_config["collision_check_steps"])
        )
        collision_free, reason = joint_path_collision_free(
            world.robot, pre_to_grasp, obstacles, allowed_contact_body_ids=allowed_contacts
        )
        if not collision_free:
            _reject(candidate, f"approach_{reason}")
            continue
        grasp_to_lift = interpolate_joints(
            grasp_ik.joints, lift_ik.joints, int(filter_config["collision_check_steps"])
        )
        collision_free, reason = joint_path_collision_free(
            world.robot, grasp_to_lift, obstacles, allowed_contact_body_ids=allowed_contacts
        )
        if not collision_free:
            _reject(candidate, f"lift_{reason}")
    stage_counts["ik_and_trajectory"] = sum(candidate.accepted for candidate in candidates)
    rejection_counts: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate.rejection_reasons:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    return FilterSummary(stage_counts, rejection_counts)
