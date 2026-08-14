import numpy as np

from open_vocab_grasping.grasping.contact import (
    closing_axis_center_shift,
    ContactQuality,
    contact_support_decision,
    parallel_jaw_symmetric_rotation,
    tabletop_topdown_rotation,
    parallel_jaw_contact_quality,
)
from open_vocab_grasping.grasping.interface import GraspCandidate


def candidate() -> GraspCandidate:
    return GraspCandidate(
        center=np.zeros(3),
        rotation=np.eye(3),
        width=0.06,
        depth=0.03,
        grasp_score=0.8,
    )


def test_bilateral_target_support_scores_above_single_wall() -> None:
    rng = np.random.default_rng(7)
    common_xz = rng.uniform([-0.008, -0.015], [0.008, 0.015], size=(80, 2))
    both_sides = np.column_stack(
        (np.tile(common_xz[:, 0], 2), np.repeat([-0.02, 0.02], 80), np.tile(common_xz[:, 1], 2))
    )
    single_wall = np.column_stack((common_xz[:, 0], np.full(80, 0.02), common_xz[:, 1]))

    bilateral = parallel_jaw_contact_quality(
        candidate(), both_sides, gripper_height_m=0.03, contact_depth_m=0.05,
        minimum_side_offset_m=0.003, minimum_points=12,
    )
    unilateral = parallel_jaw_contact_quality(
        candidate(), single_wall, gripper_height_m=0.03, contact_depth_m=0.05,
        minimum_side_offset_m=0.003, minimum_points=12,
    )

    assert bilateral.balance == 1.0
    assert bilateral.score > 0.5
    assert unilateral.balance == 0.0
    assert unilateral.score == 0.0


def test_contact_quality_requires_enough_points() -> None:
    sparse = np.array([[0.0, -0.02, 0.0], [0.0, 0.02, 0.0]])
    quality = parallel_jaw_contact_quality(
        candidate(), sparse, gripper_height_m=0.03, contact_depth_m=0.05,
        minimum_side_offset_m=0.003, minimum_points=12,
    )
    assert quality.point_count == 2
    assert quality.score == 0.0


def test_closing_axis_refinement_recenters_offset_target() -> None:
    points = np.array(
        [[0.0, y, z] for y in np.linspace(0.005, 0.045, 30) for z in (-0.01, 0.01)]
    )
    shift = closing_axis_center_shift(
        candidate(), points, gripper_height_m=0.03, contact_depth_m=0.05,
        maximum_shift_m=0.025,
    )
    assert np.isclose(shift, 0.025)


def test_centered_topdown_candidate_can_use_single_view_occlusion_fallback() -> None:
    quality = ContactQuality(score=0.01, balance=0.02, enclosure=0.5, point_count=180)
    accepted, mode = contact_support_decision(
        quality,
        minimum_score=0.05,
        semantic_center_distance_m=0.006,
        downward_alignment=0.95,
        occlusion_fallback_max_center_distance_m=0.012,
        occlusion_fallback_min_alignment=0.90,
        occlusion_fallback_min_points=100,
    )
    assert accepted
    assert mode == "centered_topdown_single_view_fallback"


def test_occlusion_fallback_rejects_off_center_candidate() -> None:
    quality = ContactQuality(score=0.01, balance=0.02, enclosure=0.5, point_count=180)
    accepted, mode = contact_support_decision(
        quality,
        minimum_score=0.05,
        semantic_center_distance_m=0.04,
        downward_alignment=0.95,
        occlusion_fallback_max_center_distance_m=0.012,
        occlusion_fallback_min_alignment=0.90,
        occlusion_fallback_min_points=100,
    )
    assert not accepted
    assert mode == "insufficient_support"


def test_parallel_jaw_symmetry_preserves_approach_and_valid_rotation() -> None:
    rotation = np.eye(3)
    symmetric = parallel_jaw_symmetric_rotation(rotation)
    assert np.allclose(symmetric[:, 2], rotation[:, 2])
    assert np.allclose(symmetric[:, :2], -rotation[:, :2])
    assert np.isclose(np.linalg.det(symmetric), 1.0)


def test_tabletop_refinement_preserves_closing_yaw_and_points_down() -> None:
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    refined = tabletop_topdown_rotation(rotation)
    assert refined is not None
    np.testing.assert_allclose(refined[:, 2], [0.0, 0.0, -1.0])
    original_closing_xy = rotation[:2, 1] / np.linalg.norm(rotation[:2, 1])
    np.testing.assert_allclose(refined[:2, 1], original_closing_xy)
    assert np.isclose(np.linalg.det(refined), 1.0)
