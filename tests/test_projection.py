import numpy as np

from open_vocab_grasping.geometry.projection import backproject_pixels, camera_intrinsics, project_points


def test_projection_backprojection_roundtrip() -> None:
    intrinsic = camera_intrinsics(640, 480, 60.0)
    points = np.array([[0.0, 0.0, 0.8], [0.15, -0.08, 1.2], [-0.2, 0.1, 1.5]])
    pixels, depth = project_points(points, intrinsic)
    recovered = backproject_pixels(pixels, depth, intrinsic)
    np.testing.assert_allclose(recovered, points, atol=1e-12)


def test_principal_point_projects_center_ray() -> None:
    intrinsic = camera_intrinsics(320, 240, 60.0)
    pixel, _ = project_points(np.array([[0.0, 0.0, 1.0]]), intrinsic)
    np.testing.assert_allclose(pixel[0], intrinsic[:2, 2], atol=1e-12)

