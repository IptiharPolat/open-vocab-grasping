from __future__ import annotations

import numpy as np


def camera_intrinsics(width: int, height: int, vertical_fov_deg: float) -> np.ndarray:
    fy = 0.5 * float(height) / np.tan(np.deg2rad(vertical_fov_deg) / 2.0)
    fx = fy
    return np.array(
        [[fx, 0.0, (width - 1.0) / 2.0], [0.0, fy, (height - 1.0) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Projection requires positive camera-frame z depth")
    uvw = (np.asarray(intrinsic, dtype=np.float64) @ points.T).T
    return uvw[:, :2] / uvw[:, 2:3], points[:, 2].copy()


def backproject_pixels(uv: np.ndarray, depth_m: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pixels = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    if len(pixels) != len(depth):
        raise ValueError("uv and depth must have the same number of samples")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (pixels[:, 0] - cx) * depth / fx
    y = (pixels[:, 1] - cy) * depth / fy
    return np.column_stack((x, y, depth))

