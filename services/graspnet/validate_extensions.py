"""Run minimal GPU computations through GraspNet's compiled extensions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="third_party/graspnet-baseline")
    args = parser.parse_args()

    root = Path(args.baseline_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "pointnet2"))
    sys.path.insert(0, str(root / "knn"))

    import torch
    from knn_modules import knn
    from pointnet2_utils import furthest_point_sample

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; extension validation must run on the GPU host")

    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
        dtype=torch.float32,
        device="cuda",
    )
    fps_indices = furthest_point_sample(points.contiguous(), 2)

    # KNN expects (batch, dimensions, points) and returns one-based indices.
    reference = points.transpose(1, 2).contiguous()
    query = points[:, :1].transpose(1, 2).contiguous()
    knn_indices = knn(reference, query, k=1)

    result = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "pointnet2_fps_indices": fps_indices.cpu().tolist(),
        "knn_indices_one_based": knn_indices.cpu().tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
