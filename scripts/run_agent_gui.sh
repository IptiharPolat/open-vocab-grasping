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
if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set. Start this command from the graphical Ubuntu/VS Code desktop." >&2
  exit 2
fi

echo "PyBullet GUI will open after the Agent plan passes validation."
exec "${OVG_PYTHON}" -m open_vocab_grasping.cli agent \
  --config configs/agent_gui.yaml "$@"
