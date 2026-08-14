from __future__ import annotations

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    transformed = (np.asarray(transform) @ homogeneous.T).T[:, :3]
    return transformed.reshape(points.shape)


def look_at_T_world_camera(
    eye_world: np.ndarray, target_world: np.ndarray, up_world: np.ndarray
) -> np.ndarray:
    """Return CV-camera pose: +x right, +y down, +z forward, in world ENU."""
    eye = np.asarray(eye_world, dtype=np.float64)
    forward = np.asarray(target_world, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    up_hint = np.asarray(up_world, dtype=np.float64)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    camera_up /= np.linalg.norm(camera_up)
    rotation_world_camera = np.column_stack((right, -camera_up, forward))
    return make_transform(rotation_world_camera, eye)


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized PyBullet xyzw quaternion."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                          (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s,
                          0.25 * s, (r[1, 0] - r[0, 1]) / s])
    return q / np.linalg.norm(q)

