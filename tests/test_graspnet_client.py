from pathlib import Path

import numpy as np

from open_vocab_grasping.grasping.graspnet_client import (
    R_GRASPNET_GRASP_TOOL,
    GraspNetFileClient,
    camera_grasps_to_world_tool,
)


def test_graspnet_request_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "request.npz"
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.ones((3, 4), dtype=np.float32)
    k = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    GraspNetFileClient.write_request(path, rgb, depth, k, np.ones_like(depth, dtype=bool))
    request = GraspNetFileClient.read_request(path)
    assert request["rgb"].shape == (3, 4, 3)
    assert request["depth_m"].dtype == np.float32


def test_graspnet_response_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "response.npz"
    np.savez_compressed(path, schema_version=np.array("1.0"),
                       centers=np.array([[0.1, 0.2, 0.3]]), rotations=np.eye(3)[None],
                       widths=np.array([0.05]), depths=np.array([0.02]), scores=np.array([0.9]),
                       collision=np.array([False]))
    candidates = GraspNetFileClient.read_response(path)
    assert len(candidates) == 1
    assert candidates[0].frame == "camera"
    assert candidates[0].grasp_score == 0.9


def test_graspnet_camera_to_world_tool_axis_adapter(tmp_path: Path) -> None:
    path = tmp_path / "response.npz"
    np.savez_compressed(
        path,
        schema_version=np.array("1.0"),
        centers=np.array([[0.1, 0.2, 0.3]]),
        rotations=np.eye(3)[None],
        widths=np.array([0.05]),
        depths=np.array([0.02]),
        scores=np.array([0.9]),
        collision=np.array([False]),
    )
    candidate = GraspNetFileClient.read_response(path)[0]
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [1.0, 2.0, 3.0]
    transformed = camera_grasps_to_world_tool([candidate], T_world_camera)[0]
    assert transformed.frame == "world"
    assert np.allclose(transformed.center, [1.12, 2.2, 3.3])
    assert np.allclose(transformed.rotation, R_GRASPNET_GRASP_TOOL)
    # Tool +z is GraspNet +x (approach); Panda tool +y is GraspNet +y (closing).
    assert np.allclose(transformed.rotation[:, 2], [1.0, 0.0, 0.0])
    assert np.allclose(transformed.rotation[:, 1], [0.0, 1.0, 0.0])
    assert np.allclose(transformed.rotation[:, 0], [0.0, 0.0, -1.0])
    assert np.isclose(np.linalg.det(transformed.rotation), 1.0)
