from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class SceneObject:
    name: str
    body_id: int
    color_name: str
    nominal_size: tuple[float, float, float]

    @property
    def label(self) -> str:
        return f"{self.color_name} {self.name}"


COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.80, 0.08, 0.06, 1.0),
    "blue": (0.05, 0.18, 0.82, 1.0),
    "green": (0.06, 0.62, 0.18, 1.0),
    "yellow": (0.90, 0.65, 0.04, 1.0),
    "brown": (0.52, 0.28, 0.10, 1.0),
}


def _create_primitive(
    client_id: int,
    name: str,
    position: np.ndarray,
    yaw: float,
    color: tuple[float, float, float, float],
    use_mesh_assets: bool = False,
    mug_mesh_scale: float = 0.90,
    bowl_mesh_scale: float = 0.68,
) -> tuple[int, tuple[float, float, float]]:
    orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw])
    if name == "mug" and use_mesh_assets:
        # PyBullet's mug mesh has a 0.082 m body diameter and 0.100 m height at
        # unit scale. Keep the body inside Panda's 0.080 m opening and match the
        # collision proxy to the visible body instead of the handle-inclusive AABB.
        scale = float(mug_mesh_scale)
        if not 0.5 <= scale <= 1.0:
            raise ValueError("scene.mug_mesh_scale must be between 0.5 and 1.0")
        body_radius = 0.041 * scale
        height = 0.100 * scale
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=body_radius, height=height, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName="objects/mug.obj",
            meshScale=[scale, scale, scale],
            visualFramePosition=[0.0, 0.0, -height / 2.0],
            rgbaColor=color,
            physicsClientId=client_id,
        )
        body = p.createMultiBody(
            baseMass=0.06,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(position[0]), float(position[1]), height / 2.0 + 0.002],
            baseOrientation=orientation,
            physicsClientId=client_id,
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.01,
            rollingFriction=0.001,
            restitution=0.0,
            physicsClientId=client_id,
        )
        size = (0.082 * scale, 0.121633 * scale, height)
        return body, size
    if name == "bowl" and use_mesh_assets:
        # Project-owned deterministic mesh gives a real round rim and cavity;
        # a cylinder proxy keeps contact physics deterministic.
        scale = float(bowl_mesh_scale)
        if not 0.5 <= scale <= 1.0:
            raise ValueError("scene.bowl_mesh_scale must be between 0.5 and 1.0")
        radius, height = 0.055 * scale, 0.055 * scale
        project_root = Path(__file__).resolve().parents[3]
        mesh_path = project_root / "assets" / "procedural" / "bowl.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Procedural bowl mesh missing at {mesh_path}; run scripts/generate_procedural_assets.py"
            )
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_MESH,
            fileName=str(mesh_path),
            meshScale=[scale, scale, scale],
            visualFramePosition=[0.0, 0.0, -height / 2.0],
            rgbaColor=color,
            physicsClientId=client_id,
        )
        body = p.createMultiBody(
            baseMass=0.06,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(position[0]), float(position[1]), height / 2.0 + 0.002],
            baseOrientation=orientation,
            physicsClientId=client_id,
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.01,
            rollingFriction=0.001,
            restitution=0.0,
            physicsClientId=client_id,
        )
        return body, (2 * radius, 2 * radius, height)
    if name == "box" and use_mesh_assets:
        # A project-owned cardboard parcel assembled from primitive visual
        # shapes. Tape and a light shipping label give YOLO-World semantic cues
        # without downloading an unlicensed texture.
        half = np.array([0.040, 0.030, 0.030], dtype=np.float64)
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half.tolist(), physicsClientId=client_id
        )
        identity = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        visual = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_BOX] * 9,
            halfExtents=[
                half.tolist(),
                [0.006, 0.0305, 0.001],
                [0.0405, 0.006, 0.001],
                [0.006, 0.001, 0.030],
                [0.006, 0.001, 0.030],
                [0.017, 0.001, 0.010],
                [0.017, 0.001, 0.010],
                [0.001, 0.017, 0.010],
                [0.001, 0.017, 0.010],
            ],
            visualFramePositions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, float(half[2] + 0.001)],
                [0.0, 0.0, float(half[2] + 0.0015)],
                [0.0, float(-half[1] - 0.001), 0.0],
                [0.0, float(half[1] + 0.001), 0.0],
                [0.015, float(-half[1] - 0.002), 0.004],
                [-0.015, float(half[1] + 0.002), 0.004],
                [float(-half[0] - 0.002), 0.010, 0.004],
                [float(half[0] + 0.002), -0.010, 0.004],
            ],
            visualFrameOrientations=[identity] * 9,
            rgbaColors=[
                color,
                [0.16, 0.10, 0.05, 1.0],
                [0.16, 0.10, 0.05, 1.0],
                [0.16, 0.10, 0.05, 1.0],
                [0.16, 0.10, 0.05, 1.0],
                [0.92, 0.90, 0.78, 1.0],
                [0.92, 0.90, 0.78, 1.0],
                [0.92, 0.90, 0.78, 1.0],
                [0.92, 0.90, 0.78, 1.0],
            ],
            physicsClientId=client_id,
        )
        body = p.createMultiBody(
            baseMass=0.06,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[float(position[0]), float(position[1]), float(half[2] + 0.002)],
            baseOrientation=orientation,
            physicsClientId=client_id,
        )
        p.changeDynamics(
            body,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.01,
            rollingFriction=0.001,
            restitution=0.0,
            physicsClientId=client_id,
        )
        return body, tuple((2.0 * half).tolist())
    if name == "box":
        half = [0.022, 0.022, 0.025]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client_id)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=client_id)
        size = (0.044, 0.044, 0.050)
    elif name == "bottle":
        radius, height = 0.022, 0.105
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=client_id
        )
        size = (2 * radius, 2 * radius, height)
    elif name == "bowl":
        radius, height = 0.038, 0.028
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=client_id
        )
        size = (2 * radius, 2 * radius, height)
    elif name == "mug":
        # A compact cup proxy with a clear semantic label. The PyBullet-distributed
        # mesh is intentionally deferred to the detector stage because its broad
        # AABB makes deterministic grasp smoke tests less stable.
        radius, height = 0.028, 0.065
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=client_id
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=client_id
        )
        size = (2 * radius, 2 * radius, height)
    else:
        raise ValueError(f"Unsupported scene object: {name}")
    body = p.createMultiBody(
        baseMass=0.06,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position.tolist(),
        baseOrientation=orientation,
        physicsClientId=client_id,
    )
    p.changeDynamics(
        body,
        -1,
        lateralFriction=1.2,
        spinningFriction=0.01,
        rollingFriction=0.001,
        restitution=0.0,
        physicsClientId=client_id,
    )
    return body, size


def create_scene_objects(
    client_id: int, config: dict[str, Any], rng: np.random.Generator
) -> dict[str, SceneObject]:
    names = list(config["scene"]["object_names"])[: int(config["scene"]["num_objects"])]
    slots = np.array([[0.46, -0.18], [0.57, 0.18], [0.67, -0.08], [0.38, 0.14]])
    # The default scene deliberately includes a red bottle so compound prompts
    # such as "red bottle" have a matching truth object.
    color_names = ["blue", "red", "green", "yellow"]
    result: dict[str, SceneObject] = {}
    for index, name in enumerate(names):
        xy = slots[index].copy() + rng.uniform(-0.012, 0.012, size=2)
        yaw = float(rng.uniform(-np.pi, np.pi))
        color_name = color_names[index % len(color_names)]
        if name == "box" and bool(config["scene"].get("use_mesh_assets", False)):
            color_name = "brown"
        approximate_heights = {"box": 0.050, "bottle": 0.105, "bowl": 0.028, "mug": 0.125}
        position = np.array([xy[0], xy[1], approximate_heights[name] / 2.0 + 0.002])
        body_id, size = _create_primitive(
            client_id,
            name,
            position,
            yaw,
            COLORS[color_name],
            bool(config["scene"].get("use_mesh_assets", False)),
            float(config["scene"].get("mug_mesh_scale", 0.90)),
            float(config["scene"].get("bowl_mesh_scale", 0.68)),
        )
        result[name] = SceneObject(name, body_id, color_name, size)
    return result
