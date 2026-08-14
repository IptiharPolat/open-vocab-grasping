from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from open_vocab_grasping.geometry.projection import project_points
from open_vocab_grasping.geometry.transforms import transform_points
from open_vocab_grasping.grasping.interface import GraspCandidate


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    confidence: float


@dataclass(frozen=True)
class AssociationRecord:
    candidate_index: int
    center_camera_m: tuple[float, float, float]
    projected_uv: tuple[float, float] | None
    candidate_depth_m: float | None
    reference_depth_m: float | None
    depth_delta_m: float | None
    inside_box: bool
    depth_consistent: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "center_camera_m": self.center_camera_m,
            "projected_uv": self.projected_uv,
            "candidate_depth_m": self.candidate_depth_m,
            "reference_depth_m": self.reference_depth_m,
            "depth_delta_m": self.depth_delta_m,
            "inside_box": self.inside_box,
            "depth_consistent": self.depth_consistent,
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
        }


def point_in_box(point_uv: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> bool:
    u, v = np.asarray(point_uv, dtype=np.float64)
    x1, y1, x2, y2 = box_xyxy
    return bool(x1 <= u <= x2 and y1 <= v <= y2)


def depth_consistency(
    candidate_depth_m: float,
    depth_image_m: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    tolerance_m: float = 0.05,
) -> tuple[bool, float]:
    consistent, delta, _ = depth_consistency_details(
        candidate_depth_m, depth_image_m, box_xyxy, tolerance_m
    )
    return consistent, delta


def depth_consistency_details(
    candidate_depth_m: float,
    depth_image_m: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    tolerance_m: float = 0.05,
    shrink_fraction: float = 0.10,
) -> tuple[bool, float, float]:
    height, width = depth_image_m.shape
    x1, y1, x2, y2 = box_xyxy
    dx = max(0.0, x2 - x1) * shrink_fraction
    dy = max(0.0, y2 - y1) * shrink_fraction
    x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 - dx, y2 - dy
    xa, xb = max(0, int(np.floor(x1))), min(width, int(np.ceil(x2)) + 1)
    ya, yb = max(0, int(np.floor(y1))), min(height, int(np.ceil(y2)) + 1)
    region = depth_image_m[ya:yb, xa:xb]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size == 0:
        return False, float("inf"), float("nan")
    reference = float(np.median(valid))
    delta = abs(float(candidate_depth_m) - reference)
    return delta <= tolerance_m, delta, reference


def associate_candidates(
    candidates: list[GraspCandidate],
    detection: Detection,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    T_camera_candidate_frame: np.ndarray,
    depth_tolerance_m: float = 0.05,
) -> list[GraspCandidate]:
    associated, _ = associate_candidates_with_records(
        candidates, detection, depth_m, intrinsic, T_camera_candidate_frame, depth_tolerance_m
    )
    return associated


def associate_candidates_with_records(
    candidates: list[GraspCandidate],
    detection: Detection,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    T_camera_candidate_frame: np.ndarray,
    depth_tolerance_m: float = 0.05,
    box_shrink_fraction: float = 0.10,
) -> tuple[list[GraspCandidate], list[AssociationRecord]]:
    records: list[AssociationRecord] = []
    for index, candidate in enumerate(candidates):
        center_camera = transform_points(T_camera_candidate_frame, candidate.center[None, :])[0]
        if center_camera[2] <= 0:
            candidate.accepted = False
            if "behind_camera" not in candidate.rejection_reasons:
                candidate.rejection_reasons.append("behind_camera")
            records.append(
                AssociationRecord(index, tuple(center_camera.tolist()), None, None, None, None,
                                  False, False, False, tuple(candidate.rejection_reasons))
            )
            continue
        uv, z = project_points(center_camera[None, :], intrinsic)
        inside = point_in_box(uv[0], detection.bbox_xyxy)
        consistent, delta, reference = depth_consistency_details(
            z[0], depth_m, detection.bbox_xyxy, depth_tolerance_m, box_shrink_fraction
        )
        candidate.detection_confidence = detection.confidence
        candidate.depth_inconsistency = delta
        x1, y1, x2, y2 = detection.bbox_xyxy
        box_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        diagonal = max(np.hypot(x2 - x1, y2 - y1), 1.0)
        candidate.center_score = float(max(0.0, 1.0 - np.linalg.norm(uv[0] - box_center) / diagonal))
        if not inside:
            candidate.accepted = False
            if "outside_detection_box" not in candidate.rejection_reasons:
                candidate.rejection_reasons.append("outside_detection_box")
        if not consistent:
            candidate.accepted = False
            if "depth_inconsistent" not in candidate.rejection_reasons:
                candidate.rejection_reasons.append("depth_inconsistent")
        records.append(
            AssociationRecord(
                index,
                tuple(float(value) for value in center_camera),
                tuple(float(value) for value in uv[0]),
                float(z[0]),
                reference if np.isfinite(reference) else None,
                delta if np.isfinite(delta) else None,
                inside,
                consistent,
                candidate.accepted,
                tuple(candidate.rejection_reasons),
            )
        )
    return candidates, records
