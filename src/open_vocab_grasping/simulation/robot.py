from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p
import pybullet_data


class PandaRobot:
    arm_joints = tuple(range(7))
    finger_joints = (9, 10)
    end_effector_link = 11

    def __init__(self, client_id: int, config: dict[str, Any]):
        self.client_id = client_id
        self.config = config["robot"]
        self.body_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=client_id,
        )
        self.lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.joint_ranges = self.upper_limits - self.lower_limits
        self.home = np.asarray(self.config["home_joints"], dtype=np.float64)
        for finger_link in self.finger_joints:
            p.changeDynamics(
                self.body_id,
                finger_link,
                lateralFriction=float(self.config.get("finger_lateral_friction", 2.0)),
                spinningFriction=float(self.config.get("finger_spinning_friction", 0.01)),
                restitution=0.0,
                physicsClientId=self.client_id,
            )
        self.reset_home()

    def reset_home(self) -> None:
        for joint, value in zip(self.arm_joints, self.home):
            p.resetJointState(self.body_id, joint, float(value), physicsClientId=self.client_id)
        for joint in self.finger_joints:
            p.resetJointState(self.body_id, joint, 0.04, physicsClientId=self.client_id)

    def joint_positions(self) -> np.ndarray:
        return np.array(
            [p.getJointState(self.body_id, joint, physicsClientId=self.client_id)[0] for joint in self.arm_joints]
        )

    def set_arm_targets(self, targets: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.body_id,
            self.arm_joints,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(targets).tolist(),
            forces=[float(self.config["joint_force_n"])] * 7,
            positionGains=[0.08] * 7,
            velocityGains=[1.0] * 7,
            physicsClientId=self.client_id,
        )

    def set_gripper(self, width_m: float) -> None:
        per_finger = float(np.clip(width_m / 2.0, 0.0, 0.04))
        p.setJointMotorControlArray(
            self.body_id,
            self.finger_joints,
            p.POSITION_CONTROL,
            targetPositions=[per_finger, per_finger],
            forces=[float(self.config["finger_force_n"])] * 2,
            positionGains=[0.2, 0.2],
            physicsClientId=self.client_id,
        )

    def end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self.body_id, self.end_effector_link, computeForwardKinematics=True, physicsClientId=self.client_id
        )
        return np.asarray(state[4]), np.asarray(state[5])
