from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from open_vocab_grasping.grasping.interface import GraspCandidate


def parallel_jaw_symmetric_rotation(rotation: np.ndarray) -> np.ndarray:
    """Swap the two identical fingers while preserving the approach axis."""
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    return value @ np.diag([-1.0, -1.0, 1.0])


def tabletop_topdown_rotation(rotation: np.ndarray) -> np.ndarray | None:
    """Preserve predicted horizontal closing yaw while enforcing top-down approach."""
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    closing = value[:, 1].copy()
    closing[2] = 0.0
    norm = float(np.linalg.norm(closing))
    if norm < 1e-6:
        return None
    y_axis = closing / norm
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


@dataclass(frozen=True)
class ContactQuality:
    """Visible target support inside a parallel-jaw closing volume.

    Candidate rotations use Panda tool axes expressed in world coordinates:
    tool x is gripper height, tool y is the finger closing direction and tool z
    is the approach direction.  This metric uses only the RGB-D target cluster;
    simulator body IDs and object poses are never consulted.
    """

    score: float
    balance: float
    enclosure: float
    point_count: int


def contact_support_decision(
    quality: ContactQuality,
    *,
    minimum_score: float,
    semantic_center_distance_m: float | None,
    downward_alignment: float,
    occlusion_fallback_max_center_distance_m: float,
    occlusion_fallback_min_alignment: float,
    occlusion_fallback_min_points: int,
) -> tuple[bool, str]:
    if quality.score >= minimum_score:
        return True, "visible_bilateral_support"
    occlusion_fallback = (
        semantic_center_distance_m is not None
        and semantic_center_distance_m <= occlusion_fallback_max_center_distance_m
        and downward_alignment >= occlusion_fallback_min_alignment
        and quality.point_count >= occlusion_fallback_min_points
    )
    if occlusion_fallback:
        return True, "centered_topdown_single_view_fallback"
    return False, "insufficient_support"


def closing_axis_center_shift(
    candidate: GraspCandidate,
    target_points_world: np.ndarray,
    *,
    gripper_height_m: float,
    contact_depth_m: float,
    maximum_shift_m: float,
) -> float:
    """Return a bounded closing-axis recentering offset from visible target points."""
    points = np.asarray(target_points_world, dtype=np.float64)
    if points.size == 0 or maximum_shift_m <= 0:
        return 0.0
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_points_world must have shape (N, 3)")
    rotation = np.asarray(candidate.rotation, dtype=np.float64)
    local = (points - np.asarray(candidate.center, dtype=np.float64)) @ rotation
    # Use a wider closing-axis gate than the final contact volume so a candidate
    # slightly offset to one object wall can be corrected without changing its
    # GraspNet-predicted orientation, depth or width.
    broad_half_opening = min(0.08, max(float(candidate.width) / 2.0, 0.02) + maximum_shift_m)
    support = (
        (np.abs(local[:, 0]) <= gripper_height_m / 2.0)
        & (np.abs(local[:, 1]) <= broad_half_opening)
        & (np.abs(local[:, 2]) <= contact_depth_m / 2.0)
    )
    closing_coordinates = local[support, 1]
    if closing_coordinates.size == 0:
        return 0.0
    return float(np.clip(np.median(closing_coordinates), -maximum_shift_m, maximum_shift_m))


def parallel_jaw_contact_quality(
    candidate: GraspCandidate,
    target_points_world: np.ndarray,
    *,
    gripper_height_m: float,
    contact_depth_m: float,
    minimum_side_offset_m: float,
    minimum_points: int,
) -> ContactQuality:
    points = np.asarray(target_points_world, dtype=np.float64)
    if points.size == 0:
        return ContactQuality(0.0, 0.0, 0.0, 0)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_points_world must have shape (N, 3)")
    if gripper_height_m <= 0 or contact_depth_m <= 0 or minimum_points <= 0:
        raise ValueError("contact-volume dimensions and minimum_points must be positive")

    rotation = np.asarray(candidate.rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("candidate.rotation must have shape (3, 3)")
    # Row-vector equivalent of R_tool_world @ (p_world - center_world).
    local = (points - np.asarray(candidate.center, dtype=np.float64)) @ rotation
    half_height = gripper_height_m / 2.0
    half_depth = contact_depth_m / 2.0
    half_opening = max(float(candidate.width) / 2.0, minimum_side_offset_m)
    inside = (
        (np.abs(local[:, 0]) <= half_height)
        & (np.abs(local[:, 1]) <= half_opening + 0.005)
        & (np.abs(local[:, 2]) <= half_depth)
    )
    closing_coordinates = local[inside, 1]
    count = int(closing_coordinates.size)
    if count < minimum_points:
        return ContactQuality(0.0, 0.0, 0.0, count)

    negative = int(np.count_nonzero(closing_coordinates <= -minimum_side_offset_m))
    positive = int(np.count_nonzero(closing_coordinates >= minimum_side_offset_m))
    side_total = negative + positive
    balance = 0.0 if side_total == 0 else 2.0 * min(negative, positive) / side_total

    lower, upper = np.percentile(closing_coordinates, [10.0, 90.0])
    visible_span = max(0.0, float(upper - lower))
    expected_span = max(2.0 * minimum_side_offset_m, min(float(candidate.width), 0.08))
    span_score = float(np.clip(visible_span / expected_span, 0.0, 1.0))
    centering_scale = max(expected_span / 2.0, minimum_side_offset_m)
    centering_score = float(np.exp(-abs(float(np.median(closing_coordinates))) / centering_scale))
    enclosure = float(np.sqrt(span_score * centering_score))
    support = float(np.clip(count / max(2.0 * minimum_points, 1.0), 0.0, 1.0))
    score = float(balance * enclosure * support)
    return ContactQuality(score, float(balance), enclosure, count)
