#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/ubuntu/miniforge3/bin/conda}"
ENV_NAME="${GRASPNET_ENV_NAME:-ovg-graspnet-cu117}"

env -u PYTHONPATH "${CONDA_BIN}" env create \
  -f "${PROJECT_DIR}/services/graspnet/environment-cu117.yml"

# Install current wheels for the subset of API dependencies exercised by
# inference/collision filtering. These versions retain the NumPy 1.x ABI.
env -u PYTHONPATH "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  "trimesh<5" "transforms3d<1" "pywavefront<2" "h5py<4" \
  "scikit-image==0.21.0" "cvxopt<2" "dill<1" "grasp-nms==1.0.2" \
  "ruamel.yaml<1" "colorlog<7" "multiprocess<1" "setproctitle<2" \
  "ffmpeg-python<1" "pyserial<4"

# graspnetAPI 1.2.11 transitively requests an obsolete OpenCV build whose
# isolated build tries to compile an incompatible old NumPy on Python 3.10.
# The inference service only needs GraspGroup, so install the API itself after
# modern Open3D/OpenCV have been installed explicitly by the environment file.
env -u PYTHONPATH "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  --no-deps autolab-core==1.1.1 autolab-perception==1.0.0 graspnetAPI==1.2.11

echo "Created ${ENV_NAME}. Next: bash services/graspnet/build-extensions-cu117.sh"
