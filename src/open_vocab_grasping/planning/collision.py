from __future__ import annotations

import numpy as np
import pybullet as p

from open_vocab_grasping.simulation.robot import PandaRobot


def table_clearance_ok(center_world: np.ndarray, minimum_z: float = 0.012) -> bool:
    return bool(np.asarray(center_world)[2] >= minimum_z)


def pointcloud_clearance(
    center_world: np.ndarray,
    scene_points_world: np.ndarray,
    target_points_world: np.ndarray,
    exclusion_radius_m: float,
    minimum_clearance_m: float,
) -> tuple[bool, float]:
    """Conservative clearance from non-target scene points around a grasp center."""
    center = np.asarray(center_world, dtype=np.float64)
    scene = np.asarray(scene_points_world, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points_world, dtype=np.float64).reshape(-1, 3)
    if len(scene) == 0:
        return True, float("inf")
    if len(target):
        low = target.min(axis=0) - float(exclusion_radius_m)
        high = target.max(axis=0) + float(exclusion_radius_m)
        outside_target_region = np.any((scene < low) | (scene > high), axis=1)
        obstacles = scene[outside_target_region]
    else:
        obstacles = scene
    if len(obstacles) == 0:
        return True, float("inf")
    clearance = float(np.min(np.linalg.norm(obstacles - center, axis=1)))
    return clearance >= minimum_clearance_m, clearance


def joint_path_collision_free(
    robot: PandaRobot,
    path: np.ndarray,
    obstacle_ids: list[int],
    allowed_target_id: int | None = None,
    allowed_contact_body_ids: set[int] | None = None,
    check_self_collision: bool = True,
) -> tuple[bool, str | None]:
    original = robot.joint_positions()
    allowed_ids = set(allowed_contact_body_ids or set())
    if allowed_target_id is not None:
        allowed_ids.add(int(allowed_target_id))
    try:
        for waypoint in path:
            for joint, value in zip(robot.arm_joints, waypoint):
                p.resetJointState(robot.body_id, joint, float(value), physicsClientId=robot.client_id)
            p.performCollisionDetection(physicsClientId=robot.client_id)
            if check_self_collision:
                self_contacts = p.getContactPoints(
                    robot.body_id, robot.body_id, physicsClientId=robot.client_id
                )
                meaningful = [
                    contact for contact in self_contacts
                    if abs(int(contact[3]) - int(contact[4])) > 1
                    and {int(contact[3]), int(contact[4])} != {6, 8}
                ]
                if meaningful:
                    return False, "self_collision"
            for obstacle in obstacle_ids:
                if obstacle in allowed_ids:
                    continue
                contacts = p.getContactPoints(robot.body_id, obstacle, physicsClientId=robot.client_id)
                if contacts:
                    return False, f"collision_body_{obstacle}"
        return True, None
    finally:
        for joint, value in zip(robot.arm_joints, original):
            p.resetJointState(robot.body_id, joint, float(value), physicsClientId=robot.client_id)
