"""Deterministic CPU protocol test; this is explicitly not GraspNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()
    with np.load(Path(args.request), allow_pickle=False) as data:
        depth = np.asarray(data["depth_m"], dtype=np.float32)
        k = np.asarray(data["intrinsic"], dtype=np.float64)
        mask = np.asarray(data["workspace_mask"], dtype=bool) & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(mask)
    if len(u) == 0:
        raise ValueError("Geometric protocol baseline received no valid depth")
    z = depth[v, u]
    points = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    # One deterministic scene-centroid candidate verifies serialization only.
    center = np.median(points, axis=0, keepdims=True).astype(np.float32)
    np.savez_compressed(
        args.response,
        schema_version=np.array("1.0"),
        generator=np.array("geometric_protocol_baseline_not_graspnet"),
        centers=center,
        rotations=np.eye(3, dtype=np.float32)[None],
        widths=np.array([0.05], dtype=np.float32),
        depths=np.array([0.02], dtype=np.float32),
        scores=np.array([0.25], dtype=np.float32),
        collision=np.array([False]),
    )


if __name__ == "__main__":
    main()
