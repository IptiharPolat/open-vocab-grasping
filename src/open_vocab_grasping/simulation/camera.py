from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p

from open_vocab_grasping.geometry.projection import camera_intrinsics
from open_vocab_grasping.geometry.transforms import look_at_T_world_camera
from open_vocab_grasping.perception.depth import depth_buffer_to_meters


@dataclass(frozen=True)
class CameraObservation:
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    intrinsic: np.ndarray
    T_world_camera: np.ndarray
    near_m: float
    far_m: float


class FixedRGBDCamera:
    def __init__(self, client_id: int, config: dict[str, Any]):
        self.client_id = client_id
        self.config = config["camera"]
        self.width = int(self.config["width"])
        self.height = int(self.config["height"])
        self.near_m = float(self.config["near_m"])
        self.far_m = float(self.config["far_m"])
        self.intrinsic = camera_intrinsics(
            self.width, self.height, float(self.config["vertical_fov_deg"])
        )
        eye = np.asarray(self.config["eye_world"], dtype=np.float64)
        target = np.asarray(self.config["target_world"], dtype=np.float64)
        up = np.asarray(self.config["up_world"], dtype=np.float64)
        self.T_world_camera = look_at_T_world_camera(eye, target, up)
        self._view_matrix = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        self._projection_matrix = p.computeProjectionMatrixFOV(
            fov=float(self.config["vertical_fov_deg"]),
            aspect=float(self.width) / float(self.height),
            nearVal=self.near_m,
            farVal=self.far_m,
        )

    def capture(self) -> CameraObservation:
        _, _, rgba, depth_buffer, segmentation = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client_id,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)[..., :3]
        depth = np.asarray(depth_buffer).reshape(self.height, self.width)
        segmentation_array = np.asarray(segmentation, dtype=np.int64).reshape(self.height, self.width)
        return CameraObservation(
            rgb=rgb,
            depth_m=depth_buffer_to_meters(depth, self.near_m, self.far_m),
            segmentation=segmentation_array,
            intrinsic=self.intrinsic.copy(),
            T_world_camera=self.T_world_camera.copy(),
            near_m=self.near_m,
            far_m=self.far_m,
        )


def segmentation_body_ids(segmentation: np.ndarray) -> np.ndarray:
    result = np.asarray(segmentation, dtype=np.int64).copy()
    foreground = result >= 0
    result[foreground] &= (1 << 24) - 1
    return result

