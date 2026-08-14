from __future__ import annotations

import numpy as np


def depth_buffer_to_meters(depth_buffer: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    """Invert OpenGL's non-linear [0, 1] z-buffer into optical-axis meters."""
    if not (0.0 < near_m < far_m):
        raise ValueError(f"Expected 0 < near < far, got near={near_m}, far={far_m}")
    buffer = np.asarray(depth_buffer, dtype=np.float64)
    return (far_m * near_m) / (far_m - (far_m - near_m) * buffer)


def meters_to_depth_buffer(depth_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float64)
    return (far_m - far_m * near_m / depth) / (far_m - near_m)

