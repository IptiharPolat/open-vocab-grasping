import numpy as np

from open_vocab_grasping.grasping.interface import GraspCandidate
from open_vocab_grasping.grasping.ranking import rank_candidates


def make_candidate(score: float, detection: float, reachable: bool = True) -> GraspCandidate:
    return GraspCandidate(np.zeros(3), np.eye(3), 0.04, 0.03, score,
                          detection_confidence=detection, reachable=reachable)


def test_ranking_penalizes_unreachable_candidate() -> None:
    candidates = [make_candidate(1.0, 1.0, False), make_candidate(0.5, 0.9, True)]
    weights = {"w_grasp": 0.4, "w_detection": 0.2, "w_center": 0.15,
               "w_clearance": 0.15, "w_depth": 0.1, "w_motion": 0.1,
               "unreachable_penalty": 2.0}
    ranked = rank_candidates(candidates, weights)
    assert ranked[0].reachable


def test_ranking_penalizes_semantic_center_distance() -> None:
    far = make_candidate(0.5, 0.9)
    near = make_candidate(0.5, 0.9)
    far.semantic_center_distance_m = 0.04
    near.semantic_center_distance_m = 0.004
    weights = {
        "w_grasp": 0.4,
        "w_detection": 0.2,
        "w_center": 0.15,
        "w_clearance": 0.15,
        "w_depth": 0.1,
        "w_motion": 0.1,
        "w_semantic_distance": 0.2,
        "semantic_distance_scale_m": 0.04,
        "unreachable_penalty": 2.0,
    }

    ranked = rank_candidates([far, near], weights)

    assert ranked[0] is near
    assert near.final_score > far.final_score
