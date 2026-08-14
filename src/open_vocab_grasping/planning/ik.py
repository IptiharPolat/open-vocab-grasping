from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from open_vocab_grasping.geometry.transforms import rotation_matrix_to_quaternion_xyzw
from open_vocab_grasping.simulation.robot import PandaRobot


@dataclass(frozen=True)
class IKResult:
    reachable: bool
    joints: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    reason: str | None = None


def solve_ik(
    robot: PandaRobot,
    position_world: np.ndarray,
    rotation_world_tool: np.ndarray,
    position_tolerance_m: float = 0.025,
    orientation_tolerance_rad: float = 0.35,
) -> IKResult:
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation_world_tool)
    solution = p.calculateInverseKinematics(
        robot.body_id,
        robot.end_effector_link,
        np.asarray(position_world).tolist(),
        quaternion.tolist(),
        lowerLimits=robot.lower_limits.tolist(),
        upperLimits=robot.upper_limits.tolist(),
        jointRanges=robot.joint_ranges.tolist(),
        restPoses=robot.home.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-5,
        physicsClientId=robot.client_id,
    )
    joints = np.asarray(solution[:7], dtype=np.float64)
    if not np.all(np.isfinite(joints)) or np.any(joints < robot.lower_limits - 1e-4) or np.any(joints > robot.upper_limits + 1e-4):
        return IKResult(False, joints, float("inf"), float("inf"), "joint_limits")
    old = robot.joint_positions()
    for joint, value in zip(robot.arm_joints, joints):
        p.resetJointState(robot.body_id, joint, float(value), physicsClientId=robot.client_id)
    actual_position, actual_quaternion = robot.end_effector_pose()
    for joint, value in zip(robot.arm_joints, old):
        p.resetJointState(robot.body_id, joint, float(value), physicsClientId=robot.client_id)
    position_error = float(np.linalg.norm(actual_position - position_world))
    desired_q = quaternion / np.linalg.norm(quaternion)
    actual_q = actual_quaternion / np.linalg.norm(actual_quaternion)
    orientation_error = float(2.0 * np.arccos(np.clip(abs(np.dot(desired_q, actual_q)), -1.0, 1.0)))
    reachable = position_error <= position_tolerance_m and orientation_error <= orientation_tolerance_rad
    return IKResult(
        reachable,
        joints,
        position_error,
        orientation_error,
        None if reachable else "pose_residual",
    )

