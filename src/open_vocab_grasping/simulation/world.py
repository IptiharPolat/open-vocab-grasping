from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from open_vocab_grasping.simulation.camera import FixedRGBDCamera
from open_vocab_grasping.simulation.objects import SceneObject, create_scene_objects
from open_vocab_grasping.simulation.robot import PandaRobot


@dataclass
class SimulationWorld:
    config: dict[str, Any]
    client_id: int
    robot: PandaRobot
    camera: FixedRGBDCamera
    table_id: int
    objects: dict[str, SceneObject]

    @classmethod
    def create(cls, config: dict[str, Any], seed: int | None = None) -> "SimulationWorld":
        gui = bool(config["simulation"].get("gui", False))
        client_id = p.connect(p.GUI if gui else p.DIRECT)
        if client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.resetSimulation(physicsClientId=client_id)
        p.setTimeStep(float(config["simulation"]["time_step"]), physicsClientId=client_id)
        p.setGravity(*config["simulation"]["gravity"], physicsClientId=client_id)
        p.setPhysicsEngineParameter(
            deterministicOverlappingPairs=1,
            fixedTimeStep=float(config["simulation"]["time_step"]),
            numSolverIterations=100,
            physicsClientId=client_id,
        )
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.48, 0.52, 0.025], physicsClientId=client_id
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.48, 0.52, 0.025],
            rgbaColor=[0.54, 0.36, 0.20, 1.0],
            physicsClientId=client_id,
        )
        table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.52, 0.0, -0.025],
            physicsClientId=client_id,
        )
        robot = PandaRobot(client_id, config)
        rng = np.random.default_rng(config.get("seed", 0) if seed is None else seed)
        objects = create_scene_objects(client_id, config, rng)
        camera = FixedRGBDCamera(client_id, config)
        world = cls(config, client_id, robot, camera, table_id, objects)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.25,
                cameraYaw=52.0,
                cameraPitch=-38.0,
                cameraTargetPosition=[0.48, 0.0, 0.12],
                physicsClientId=client_id,
            )
            for scene_object in objects.values():
                position, _ = p.getBasePositionAndOrientation(
                    scene_object.body_id, physicsClientId=client_id
                )
                p.addUserDebugText(
                    scene_object.label,
                    [0.0, 0.0, scene_object.nominal_size[2] / 2 + 0.035],
                    textColorRGB=[0.1, 0.1, 0.1],
                    textSize=1.2,
                    parentObjectUniqueId=scene_object.body_id,
                    physicsClientId=client_id,
                )
        world.step(int(config["simulation"]["settle_steps"]))
        return world

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client_id)
            if bool(self.config["simulation"].get("gui", False)) and bool(
                self.config["simulation"].get("realtime_pacing", True)
            ):
                time.sleep(float(self.config["simulation"]["time_step"]))

    def add_grasp_axes(self, center: np.ndarray, rotation: np.ndarray, length: float = 0.08) -> None:
        """Draw candidate axes: red x/closing, green y, blue z/approach."""
        origin = np.asarray(center, dtype=np.float64)
        for axis, color in zip(np.asarray(rotation).T, ([1, 0, 0], [0, 0.8, 0], [0, 0.2, 1])):
            p.addUserDebugLine(
                origin.tolist(),
                (origin + axis * length).tolist(),
                lineColorRGB=color,
                lineWidth=4.0,
                lifeTime=0,
                physicsClientId=self.client_id,
            )

    def object_position(self, name: str) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(
            self.objects[name].body_id, physicsClientId=self.client_id
        )
        return np.asarray(position, dtype=np.float64)

    def object_aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        minimum, maximum = p.getAABB(self.objects[name].body_id, physicsClientId=self.client_id)
        return np.asarray(minimum), np.asarray(maximum)

    def close(self) -> None:
        if p.isConnected(self.client_id):
            hold_s = float(self.config["simulation"].get("hold_after_run_s", 0.0))
            if bool(self.config["simulation"].get("gui", False)) and hold_s > 0:
                deadline = time.monotonic() + hold_s
                while p.isConnected(self.client_id) and time.monotonic() < deadline:
                    p.stepSimulation(physicsClientId=self.client_id)
                    time.sleep(float(self.config["simulation"]["time_step"]))
            p.disconnect(self.client_id)

    def __enter__(self) -> "SimulationWorld":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
