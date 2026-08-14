#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
exec /home/ubuntu/miniforge3/bin/conda run -n ovg python -m open_vocab_grasping.cli \
  run --target mug --seed 0 --config configs/gui_demo.yaml "$@"
