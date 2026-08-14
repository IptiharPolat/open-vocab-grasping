"""Validate upstream demo RGB-D calibration and point construction without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="third_party/graspnet-baseline")
    parser.add_argument(
        "--request-output",
        type=Path,
        help="Optionally write the official example as a schema-1.0 inference request",
    )
    args = parser.parse_args()
    data = Path(args.baseline_root) / "doc" / "example_data"
    rgb = np.asarray(Image.open(data / "color.png"))
    depth_raw = np.asarray(Image.open(data / "depth.png"))
    workspace = np.asarray(Image.open(data / "workspace_mask.png"), dtype=bool)
    meta = loadmat(data / "meta.mat")
    k = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    factor = float(np.asarray(meta["factor_depth"]).squeeze())
    depth_m = depth_raw.astype(np.float32) / factor
    valid = workspace & (depth_raw > 0)
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    points = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z))
    report = {
        "status": "official_example_input_validated_no_inference",
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth_raw.shape),
        "factor_depth": factor,
        "valid_workspace_points": int(len(points)),
        "finite_points": bool(np.all(np.isfinite(points))),
        "depth_range_m": [float(points[:, 2].min()), float(points[:, 2].max())],
    }
    if args.request_output is not None:
        args.request_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.request_output,
            schema_version=np.array("1.0"),
            rgb=rgb.astype(np.uint8),
            depth_m=depth_m.astype(np.float32),
            intrinsic=k.astype(np.float64),
            workspace_mask=valid,
        )
        report["request_output"] = str(args.request_output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
