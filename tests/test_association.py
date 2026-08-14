import numpy as np

from open_vocab_grasping.grasping.association import (
    Detection,
    associate_candidates_with_records,
    depth_consistency,
    point_in_box,
)
from open_vocab_grasping.grasping.interface import GraspCandidate


def test_point_in_detection_box() -> None:
    assert point_in_box(np.array([50.0, 60.0]), (10.0, 20.0, 90.0, 100.0))
    assert not point_in_box(np.array([9.9, 60.0]), (10.0, 20.0, 90.0, 100.0))


def test_depth_consistency_uses_box_region_median() -> None:
    depth = np.full((100, 100), 2.0)
    depth[20:41, 10:31] = 0.70
    consistent, delta = depth_consistency(0.72, depth, (10, 20, 30, 40), tolerance_m=0.03)
    assert consistent
    assert abs(delta - 0.02) < 1e-12
    inconsistent, _ = depth_consistency(0.80, depth, (10, 20, 30, 40), tolerance_m=0.03)
    assert not inconsistent


def test_association_records_accept_and_reject_reasons() -> None:
    depth = np.full((100, 100), 2.0)
    depth[20:81, 20:81] = 1.0
    intrinsic = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    candidates = [
        GraspCandidate(np.array([0.0, 0.0, 1.0]), np.eye(3), 0.04, 0.02, 0.9,
                       frame="camera"),
        GraspCandidate(np.array([1.0, 0.0, 1.0]), np.eye(3), 0.04, 0.02, 0.8,
                       frame="camera"),
    ]
    _, records = associate_candidates_with_records(
        candidates, Detection((20.0, 20.0, 80.0, 80.0), "mug", 0.7), depth,
        intrinsic, np.eye(4), depth_tolerance_m=0.03
    )
    assert records[0].accepted
    assert records[0].projected_uv == (50.0, 50.0)
    assert not records[1].accepted
    assert "outside_detection_box" in records[1].rejection_reasons
