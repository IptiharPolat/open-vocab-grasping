from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from open_vocab_grasping.evaluation.metrics import grasp_success
from open_vocab_grasping.grasping.interface import GraspCandidate
from open_vocab_grasping.planning.ik import solve_ik
from open_vocab_grasping.planning.trajectory import interpolate_joints
from open_vocab_grasping.simulation.world import SimulationWorld


class GraspState(str, enum.Enum):
    RESET = "RESET"
    OBSERVE = "OBSERVE"
    DETECT = "DETECT"
    GENERATE_GRASPS = "GENERATE_GRASPS"
    SELECT_GRASP = "SELECT_GRASP"
    MOVE_PREGRASP = "MOVE_PREGRASP"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    EVALUATE = "EVALUATE"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class ExecutionResult:
    success: bool
    final_state: GraspState
    failure_reason: str | None
    state_history: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class GraspExecutor:
    def __init__(self, world: SimulationWorld, frame_callback: Callable[[], None] | None = None):
        self.world = world
        self.config = world.config["grasp"]
        self.history: list[str] = []
        self.frame_callback = frame_callback

    def _state(self, state: GraspState) -> None:
        self.history.append(state.value)
        if self.frame_callback is not None:
            self.frame_callback()

    def _move_to(self, joints: np.ndarray) -> None:
        robot = self.world.robot
        trajectory = interpolate_joints(
            robot.joint_positions(), joints, int(self.config["trajectory_steps"])
        )
        for index, waypoint in enumerate(trajectory):
            robot.set_arm_targets(waypoint)
            self.world.step(int(self.config["control_steps_per_waypoint"]))
            if self.frame_callback is not None and index % 3 == 0:
                self.frame_callback()

    def execute(self, candidate: GraspCandidate, target_name: str) -> ExecutionResult:
        self.history = []
        robot = self.world.robot
        target_id = self.world.objects[target_name].body_id
        initial_object_position = self.world.object_position(target_name)
        self._state(GraspState.RESET)
        robot.reset_home()
        robot.set_gripper(float(self.config["max_width_m"]))
        self.world.step(60)
        self._state(GraspState.OBSERVE)
        self._state(GraspState.DETECT)
        self._state(GraspState.GENERATE_GRASPS)
        self._state(GraspState.SELECT_GRASP)

        approach_axis = candidate.rotation[:, 2]
        grasp_position = candidate.center - approach_axis * float(self.config["tool_offset_m"])
        approach_distance = (
            float(candidate.approach_distance_m)
            if candidate.approach_distance_m is not None
            else float(self.config["approach_distance_m"])
        )
        pregrasp_position = grasp_position - approach_axis * approach_distance
        pre_ik = solve_ik(robot, pregrasp_position, candidate.rotation)
        grasp_ik = solve_ik(robot, grasp_position, candidate.rotation)
        if not pre_ik.reachable or not grasp_ik.reachable:
            self._state(GraspState.FAILED)
            reason = "pregrasp_ik" if not pre_ik.reachable else "grasp_ik"
            return ExecutionResult(False, GraspState.FAILED, reason, self.history, {
                "pregrasp_position_error_m": pre_ik.position_error_m,
                "grasp_position_error_m": grasp_ik.position_error_m,
            })
        self._state(GraspState.MOVE_PREGRASP)
        self._move_to(pre_ik.joints)
        robot.set_gripper(float(self.config["max_width_m"]))
        self.world.step(40)
        self._state(GraspState.APPROACH)
        self._move_to(grasp_ik.joints)
        self._state(GraspState.CLOSE_GRIPPER)
        robot.set_gripper(0.0)
        self.world.step(int(self.config["hold_steps"]))
        self._state(GraspState.LIFT)
        lift_position = grasp_position + np.array([0.0, 0.0, float(self.config["lift_distance_m"])])
        lift_ik = solve_ik(robot, lift_position, candidate.rotation)
        if not lift_ik.reachable:
            self._state(GraspState.FAILED)
            return ExecutionResult(False, GraspState.FAILED, "lift_ik", self.history)
        self._move_to(lift_ik.joints)
        self.world.step(int(self.config["hold_steps"]))
        self._state(GraspState.EVALUATE)
        final_object_position = self.world.object_position(target_name)
        tool_position, _ = robot.end_effector_pose()
        success, metrics = grasp_success(
            initial_object_position,
            final_object_position,
            tool_position,
            float(self.config["min_lift_m"]),
        )
        final_state = GraspState.DONE if success else GraspState.FAILED
        self._state(final_state)
        return ExecutionResult(success, final_state, None if success else "object_not_lifted", self.history, metrics)
