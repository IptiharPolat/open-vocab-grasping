from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class GraspCandidate:
    center: np.ndarray
    rotation: np.ndarray
    width: float
    depth: float
    grasp_score: float
    frame: str = "world"
    collision: bool = False
    detection_confidence: float = 0.0
    center_score: float = 0.0
    clearance: float = 0.0
    depth_inconsistency: float = 0.0
    motion_cost: float = 0.0
    reachable: bool = True
    final_score: float = float("-inf")
    accepted: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    source_cluster_id: int | None = None
    semantic_center_distance_m: float | None = None
    height_axis_offset_m: float | None = None
    contact_score: float = 0.0
    contact_balance: float = 0.0
    enclosure_score: float = 0.0
    target_points_in_gripper: int = 0
    contact_center_shift_m: float = 0.0
    semantic_plane_shift_m: float = 0.0
    contact_evidence_mode: str = "not_evaluated"
    used_parallel_jaw_symmetry: bool = False
    approach_distance_m: float | None = None
    semantic_refinement_original_contact_score: float | None = None
    semantic_refinement_contact_score: float | None = None
    used_topdown_orientation_refinement: bool = False

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["center"] = self.center.tolist()
        data["rotation"] = self.rotation.tolist()
        return data


class GraspGenerator(Protocol):
    name: str

    def generate(self, *args: object, **kwargs: object) -> list[GraspCandidate]: ...
