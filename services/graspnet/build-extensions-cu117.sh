#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${GRASPNET_ENV_DIR:-/home/ubuntu/miniforge3/envs/ovg-graspnet-cu117}"
export CUDA_HOME="${ENV_DIR}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"

cd "${PROJECT_DIR}/third_party/graspnet-baseline/pointnet2"
# The upstream setup declares the extension as ``pointnet2._ext`` but the
# repository does not ship the generated package directory.  Create it before
# ``--inplace`` copies the compiled shared object.
mkdir -p pointnet2
env -u PYTHONPATH "${ENV_DIR}/bin/python" setup.py build_ext --inplace --force

cd "${PROJECT_DIR}/third_party/graspnet-baseline/knn"
mkdir -p knn_pytorch
env -u PYTHONPATH "${ENV_DIR}/bin/python" setup.py build_ext --inplace --force
