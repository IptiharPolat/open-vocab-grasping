"""Official GraspNet inference behind the versioned NPZ boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_request(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"schema_version", "rgb", "depth_m", "intrinsic", "workspace_mask"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Request missing keys: {sorted(missing)}")
        if str(data["schema_version"]) != "1.0":
            raise ValueError("Only GraspNet request schema 1.0 is supported")
        result = {key: np.asarray(data[key]) for key in required if key != "schema_version"}
    if result["rgb"].shape[:2] != result["depth_m"].shape:
        raise ValueError("RGB and depth dimensions differ")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-root", default="third_party/graspnet-baseline")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--collision-thresh", type=float, default=0.01)
    args = parser.parse_args()
    request = _load_request(Path(args.request))
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing from the isolated GraspNet environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official GraspNet inference requires its CUDA-only PointNet2 extension; "
            "torch.cuda.is_available() is false. No mock response was emitted."
        )
    root = Path(args.baseline_root).resolve()
    for subdir in ("models", "dataset", "utils", "pointnet2"):
        sys.path.insert(0, str(root / subdir))
    from graspnet import GraspNet, pred_decode
    from collision_detector import ModelFreeCollisionDetector

    depth = request["depth_m"]
    mask = request["workspace_mask"].astype(bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    k = request["intrinsic"]
    z = depth[v, u]
    points = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    rng = np.random.default_rng(0)
    choice = rng.choice(len(points), args.num_point, replace=len(points) < args.num_point)
    sampled = points[choice].astype(np.float32)
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
    net.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        prediction = pred_decode(net({"point_clouds": torch.from_numpy(sampled[None]).cuda()}))[0]
    array = prediction.detach().cpu().numpy()
    detector = ModelFreeCollisionDetector(points.astype(np.float32), voxel_size=0.01)
    from graspnetAPI import GraspGroup
    collision = detector.detect(GraspGroup(array), approach_dist=0.05, collision_thresh=args.collision_thresh)
    np.savez_compressed(
        args.response, schema_version=np.array("1.0"), centers=array[:, 13:16].astype(np.float32),
        rotations=array[:, 4:13].reshape(-1, 3, 3).astype(np.float32), widths=array[:, 1].astype(np.float32),
        depths=array[:, 3].astype(np.float32), scores=array[:, 0].astype(np.float32), collision=collision.astype(bool),
    )


if __name__ == "__main__":
    main()
