from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from open_vocab_grasping.geometry.transforms import transform_points
from open_vocab_grasping.grasping.interface import GraspCandidate


# Official GraspNet uses +x for approach, +y across the gripper opening and +z
# for gripper height.  Panda uses tool +z for approach and its URDF finger
# joints translate along tool +/-y. Columns map Panda tool axes expressed in
# the GraspNet frame: tool-x=-grasp-z, tool-y=grasp-y, tool-z=grasp-x.
R_GRASPNET_GRASP_TOOL = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


@dataclass(frozen=True)
class GraspNetInferenceMetadata:
    command: list[str]
    wall_s: float
    stdout: str
    stderr: str


class GraspNetFileClient:
    """Validated filesystem boundary for the isolated GraspNet environment."""

    name = "graspnet"

    @staticmethod
    def read_request(path: str | Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            required = {"schema_version", "rgb", "depth_m", "intrinsic", "workspace_mask"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Invalid GraspNet request; missing keys: {sorted(missing)}")
            version = str(data["schema_version"])
            rgb = np.asarray(data["rgb"])
            depth = np.asarray(data["depth_m"])
            intrinsic = np.asarray(data["intrinsic"])
            mask = np.asarray(data["workspace_mask"])
            if version != "1.0":
                raise ValueError(f"Unsupported GraspNet schema version: {version}")
            if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
                raise ValueError("RGB/depth shape mismatch in GraspNet request")
            if intrinsic.shape != (3, 3) or mask.shape != depth.shape:
                raise ValueError("Invalid intrinsic or workspace mask shape")
            if not np.all(np.isfinite(intrinsic)) or not np.any(mask & (depth > 0)):
                raise ValueError("GraspNet request has invalid calibration or no valid depth")
            return {
                "rgb": rgb.astype(np.uint8),
                "depth_m": depth.astype(np.float32),
                "intrinsic": intrinsic.astype(np.float64),
                "workspace_mask": mask.astype(bool),
            }

    @staticmethod
    def write_request(
        path: str | Path,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsic: np.ndarray,
        workspace_mask: np.ndarray,
    ) -> None:
        np.savez_compressed(
            path,
            schema_version=np.array("1.0"),
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth_m=np.asarray(depth_m, dtype=np.float32),
            intrinsic=np.asarray(intrinsic, dtype=np.float64),
            workspace_mask=np.asarray(workspace_mask, dtype=bool),
        )

    @staticmethod
    def read_response(path: str | Path) -> list[GraspCandidate]:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema_version", "centers", "rotations", "widths", "depths", "scores", "collision"
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Invalid GraspNet response; missing keys: {sorted(missing)}")
            if str(data["schema_version"]) != "1.0":
                raise ValueError(f"Unsupported GraspNet response schema: {str(data['schema_version'])}")
            count = len(data["scores"])
            if data["centers"].shape != (count, 3) or data["rotations"].shape != (count, 3, 3):
                raise ValueError("Invalid GraspNet response shape")
            for key in ("widths", "depths", "collision"):
                if data[key].shape != (count,):
                    raise ValueError(f"Invalid GraspNet response shape for {key}")
            return [
                GraspCandidate(
                    center=data["centers"][i].astype(float),
                    rotation=data["rotations"][i].astype(float),
                    width=float(data["widths"][i]),
                    depth=float(data["depths"][i]),
                    grasp_score=float(data["scores"][i]),
                    collision=bool(data["collision"][i]),
                    frame="camera",
                )
                for i in range(count)
            ]

    @staticmethod
    def run_isolated_inference(
        request_path: str | Path,
        response_path: str | Path,
        project_root: str | Path,
        config: dict[str, Any],
    ) -> tuple[list[GraspCandidate], GraspNetInferenceMetadata]:
        """Run the official model in its dedicated Conda environment."""
        root = Path(project_root).resolve()
        service_config = config["graspnet"]
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if conda is None:
            raise RuntimeError("Conda executable not found; the isolated GraspNet service cannot start")

        def project_path(value: object) -> Path:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else root / path

        command = [
            conda,
            "run",
            "-n",
            str(service_config["conda_environment"]),
            "python",
            str(project_path(service_config["inference_script"])),
            "--request",
            str(Path(request_path).resolve()),
            "--response",
            str(Path(response_path).resolve()),
            "--checkpoint",
            str(project_path(service_config["checkpoint"])),
            "--baseline-root",
            str(project_path(service_config["baseline_root"])),
            "--num-point",
            str(int(service_config.get("num_point", 20000))),
            "--collision-thresh",
            str(float(service_config.get("collision_threshold", 0.01))),
        ]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=float(service_config.get("timeout_s", 300.0)),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Official GraspNet inference exceeded {service_config.get('timeout_s', 300.0)} seconds"
            ) from exc
        wall_s = perf_counter() - started
        metadata = GraspNetInferenceMetadata(command, wall_s, completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                "Official GraspNet inference failed "
                f"(exit={completed.returncode}). stderr: {completed.stderr[-2000:]}"
            )
        return GraspNetFileClient.read_response(response_path), metadata


def camera_grasps_to_world_tool(
    candidates: list[GraspCandidate], T_world_camera: np.ndarray
) -> list[GraspCandidate]:
    """Apply ``T_world_camera @ T_camera_grasp @ T_grasp_tool`` to candidates.

    GraspNet's translation is the sampled grasp reference point.  Its official
    evaluation conversion places the parallel-jaw contact center at
    ``[depth, 0, 0]`` in the grasp frame; Panda's ``panda_grasptarget`` is this
    contact/TCP center.
    """
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("T_world_camera must have shape (4, 4)")
    R_world_camera = transform[:3, :3]
    for candidate in candidates:
        if candidate.frame != "camera":
            raise ValueError(f"Expected camera-frame GraspNet candidate, got {candidate.frame!r}")
        contact_center_camera = (
            candidate.center + candidate.rotation[:, 0] * float(candidate.depth)
        )
        candidate.center = transform_points(transform, contact_center_camera[None])[0]
        candidate.rotation = R_world_camera @ candidate.rotation @ R_GRASPNET_GRASP_TOOL
        candidate.frame = "world"
    return candidates
