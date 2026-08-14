from __future__ import annotations

from pathlib import Path

import numpy as np

from open_vocab_grasping.geometry.projection import backproject_pixels
from open_vocab_grasping.geometry.transforms import transform_points


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    T_world_camera: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv = np.column_stack((u.ravel(), v.ravel()))
    points = backproject_pixels(uv, depth_m.ravel(), intrinsic)
    valid = np.isfinite(points[:, 2]) & (points[:, 2] > 0)
    points = points[valid]
    colors = np.asarray(rgb)[..., :3].reshape(-1, 3)[valid].astype(np.float64) / 255.0
    if T_world_camera is not None:
        points = transform_points(T_world_camera, points)
    return points, colors


def crop_workspace(
    points: np.ndarray, colors: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(minimum)
    maximum = np.asarray(maximum)
    mask = np.all((points >= minimum) & (points <= maximum), axis=1)
    return points[mask], colors[mask]


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a compact ASCII PLY without requiring a display server."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
    with output.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, rgb):
            stream.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def to_open3d(points: np.ndarray, colors: np.ndarray):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return cloud

