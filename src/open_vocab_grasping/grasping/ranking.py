from __future__ import annotations

from typing import Any

import numpy as np

from open_vocab_grasping.grasping.interface import GraspCandidate


def rank_candidates(
    candidates: list[GraspCandidate], weights: dict[str, Any]
) -> list[GraspCandidate]:
    if not candidates:
        return []
    scores = np.array([candidate.grasp_score for candidate in candidates], dtype=np.float64)
    span = float(np.ptp(scores))
    normalized = (scores - scores.min()) / span if span > 1e-12 else np.ones_like(scores)
    for candidate, normalized_grasp in zip(candidates, normalized):
        semantic_distance_scale = float(weights.get("semantic_distance_scale_m", 0.05))
        if semantic_distance_scale <= 0.0:
            raise ValueError("scoring.semantic_distance_scale_m must be positive")
        semantic_distance_penalty = (
            min(candidate.semantic_center_distance_m / semantic_distance_scale, 1.0)
            if candidate.semantic_center_distance_m is not None
            else 0.0
        )
        candidate.final_score = (
            float(weights["w_grasp"]) * float(normalized_grasp)
            + float(weights["w_detection"]) * candidate.detection_confidence
            + float(weights["w_center"]) * candidate.center_score
            + float(weights["w_clearance"]) * candidate.clearance
            + float(weights.get("w_contact", 0.0)) * candidate.contact_score
            - float(weights["w_depth"]) * candidate.depth_inconsistency
            - float(weights["w_motion"]) * candidate.motion_cost
            - float(weights.get("w_semantic_distance", 0.0)) * semantic_distance_penalty
            - (0.0 if candidate.reachable else float(weights["unreachable_penalty"]))
        )
    return sorted(candidates, key=lambda candidate: (candidate.accepted, candidate.final_score), reverse=True)
