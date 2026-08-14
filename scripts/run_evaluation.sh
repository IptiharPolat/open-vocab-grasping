#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
exec /home/ubuntu/miniforge3/bin/conda run -n ovg python -m open_vocab_grasping.cli evaluate \
  --targets "mug,bottle,bowl" --episodes 30 --config configs/evaluation.yaml "$@"
