#!/usr/bin/env bash
# Start a ClearML agent worker bound to physical GPU 4 only.
set -euo pipefail

QUEUE_NAME="${CLEARML_QUEUE:-gpu4-h100-queue}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-4}"

if [[ -x "${REPO_ROOT}/.venv/bin/clearml-agent" ]]; then
  AGENT="${REPO_ROOT}/.venv/bin/clearml-agent"
elif command -v clearml-agent >/dev/null 2>&1; then
  AGENT="clearml-agent"
else
  echo "ERROR: clearml-agent not found. Run: uv pip install clearml-agent (in project .venv)" >&2
  exit 1
fi

echo "Starting ClearML agent: queue=${QUEUE_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "${AGENT}" daemon \
  --queue "${QUEUE_NAME}" \
  --gpus 1 \
  --detached
