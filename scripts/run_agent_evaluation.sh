#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

OVG_PYTHON="/home/ubuntu/miniforge3/envs/ovg/bin/python"
if [[ ! -x "${OVG_PYTHON}" ]]; then
  echo "Project interpreter not found: ${OVG_PYTHON}" >&2
  exit 1
fi

exec "${OVG_PYTHON}" -m open_vocab_grasping.cli agent-evaluate \
  --config configs/agent_graspnet.yaml \
  --suite configs/agent_instruction_suite.yaml "$@"
