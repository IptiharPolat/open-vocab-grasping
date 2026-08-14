#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# The desktop has historically exported a non-standard socks:// proxy that
# httpx rejects at import time. This local-only server needs no proxy.
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export GRADIO_ANALYTICS_ENABLED=False

OVG_PYTHON="/home/ubuntu/miniforge3/envs/ovg/bin/python"
if [[ ! -x "${OVG_PYTHON}" ]]; then
  echo "Project interpreter not found: ${OVG_PYTHON}" >&2
  echo "Create it first with: bash scripts/bootstrap.sh" >&2
  exit 1
fi

exec "${OVG_PYTHON}" -m open_vocab_grasping.cli dashboard \
  --config configs/default.yaml --host 127.0.0.1 "$@"
