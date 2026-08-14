from __future__ import annotations

import numpy as np


def interpolate_joints(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("Trajectory requires at least two waypoints")
    return np.linspace(np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64), steps)


def interpolate_positions(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    return interpolate_joints(start, goal, steps)

