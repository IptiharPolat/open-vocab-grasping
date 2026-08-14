import numpy as np

from open_vocab_grasping.geometry.transforms import invert_transform, look_at_T_world_camera, transform_points


def test_camera_world_transform_roundtrip() -> None:
    transform = look_at_T_world_camera(
        np.array([1.0, -0.8, 0.7]), np.array([0.5, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    points_camera = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 0.7]])
    world = transform_points(transform, points_camera)
    recovered = transform_points(invert_transform(transform), world)
    np.testing.assert_allclose(recovered, points_camera, atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)

