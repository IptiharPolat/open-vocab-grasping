"""Export official GraspNet candidates as headless Open3D PLY artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    import open3d as o3d
    from graspnetAPI import GraspGroup

    with np.load(args.request, allow_pickle=False) as request:
        rgb = np.asarray(request["rgb"], dtype=np.uint8)
        depth = np.asarray(request["depth_m"], dtype=np.float32)
        intrinsic = np.asarray(request["intrinsic"], dtype=np.float64)
        mask = np.asarray(request["workspace_mask"], dtype=bool) & np.isfinite(depth) & (depth > 0)
    with np.load(args.response, allow_pickle=False) as response:
        centers = np.asarray(response["centers"], dtype=np.float64)
        rotations = np.asarray(response["rotations"], dtype=np.float64)
        widths = np.asarray(response["widths"], dtype=np.float64)
        depths = np.asarray(response["depths"], dtype=np.float64)
        scores = np.asarray(response["scores"], dtype=np.float64)
        collision = np.asarray(response["collision"], dtype=bool)

    v, u = np.nonzero(mask)
    z = depth[v, u]
    points = np.column_stack(
        ((u - intrinsic[0, 2]) * z / intrinsic[0, 0],
         (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z)
    )
    colors = rgb[v, u].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    keep = np.flatnonzero(~collision)
    order = keep[np.argsort(scores[keep])[::-1]][: max(0, args.top_k)]
    grasp_array = np.column_stack(
        (
            scores[order],
            widths[order],
            np.full(len(order), 0.02),
            depths[order],
            rotations[order].reshape(-1, 9),
            centers[order],
            np.full(len(order), -1.0),
        )
    )
    geometries = GraspGroup(grasp_array).to_open3d_geometry_list() if len(order) else []
    combined = o3d.geometry.TriangleMesh()
    for geometry in geometries:
        combined += geometry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output_dir / "scene.ply"), cloud):
        raise RuntimeError("Open3D failed to write scene.ply")
    if geometries and not o3d.io.write_triangle_mesh(
        str(args.output_dir / "grippers_topk.ply"), combined, write_vertex_colors=True
    ):
        raise RuntimeError("Open3D failed to write grippers_topk.ply")
    report = {
        "source_candidate_count": int(len(scores)),
        "collision_free_count": int(len(keep)),
        "exported_count": int(len(order)),
        "indices": order.tolist(),
        "scene_ply": str((args.output_dir / "scene.ply").resolve()),
        "grippers_ply": (
            str((args.output_dir / "grippers_topk.ply").resolve()) if geometries else None
        ),
    }
    (args.output_dir / "visualization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
