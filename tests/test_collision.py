import numpy as np

from open_vocab_grasping.planning.collision import pointcloud_clearance, table_clearance_ok


def test_table_clearance_threshold() -> None:
    assert table_clearance_ok(np.array([0.5, 0.0, 0.05]), 0.035)
    assert not table_clearance_ok(np.array([0.5, 0.0, 0.02]), 0.035)


def test_pointcloud_clearance_excludes_target_but_detects_neighbour() -> None:
    center = np.array([0.5, 0.0, 0.08])
    target = center + np.array([[0.0, 0.0, 0.0], [0.02, 0.01, 0.02]])
    far_obstacle = np.array([[0.8, 0.3, 0.1]])
    clear, distance = pointcloud_clearance(center, np.vstack((target, far_obstacle)), target, 0.01, 0.06)
    assert clear
    assert distance > 0.3
    near_obstacle = np.array([[0.55, -0.03, 0.08]])
    clear, distance = pointcloud_clearance(
        center, np.vstack((target, far_obstacle, near_obstacle)), target, 0.01, 0.06
    )
    assert not clear
    assert distance < 0.06
