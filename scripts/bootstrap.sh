#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniforge3/bin/conda}"
ENV_NAME="${ENV_NAME:-ovg}"

"${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10 pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  --no-build-isolation -e "${PROJECT_DIR}[dev,yolo,ui,agent]"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  "git+https://github.com/ultralytics/CLIP.git@488e81a6711eea7346872b46ea928b367da8889d"
echo "Created Conda environment ${ENV_NAME}. Activate with: conda activate ${ENV_NAME}"
