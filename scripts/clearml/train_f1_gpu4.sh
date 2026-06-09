#!/usr/bin/env bash
# =============================================================================
# ClearML agent entrypoint — Foundation-1 3s fine-tune on a single GPU (GPU 4)
# =============================================================================
# Intended use:
#   1. Copy scripts/clearml/train_task.env.example → train_task.env and edit paths.
#   2. Install the training venv with scripts/clearml/install_training_env.sh (uv).
#   3. Register this script as a ClearML task template (see scripts/clearml/README.md).
#   4. Run clearml-agent on queue gpu4-h100-queue with CUDA_VISIBLE_DEVICES=4.
#
# The agent executes this script from the repo root (set CLONE_DIR / working dir).
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

ENV_FILE="${CLEARML_TRAIN_ENV:-${REPO_ROOT}/scripts/clearml/train_task.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
else
  echo "WARN: ${ENV_FILE} not found — using environment variables only." >&2
fi

# Pin to physical GPU 4 (only one device visible inside the worker process).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export USE_WANDB="${USE_WANDB:-0}"

# Avoid ClearML zip staging on the small root volume (default /tmp ≈ 193 GB /).
export TMPDIR="${TMPDIR:-/data/tmp}"
mkdir -p "${TMPDIR}"

: "${PRE_ENCODED_PATH:?Set PRE_ENCODED_PATH in train_task.env}"
: "${PRETRAINED_CKPT_PATH:?Set PRETRAINED_CKPT_PATH in train_task.env}"
: "${MODEL_CONFIG:=${REPO_ROOT}/models/foundation1_3s/model_config_3s.json}"
: "${SAVE_DIR:=${REPO_ROOT}/checkpoints/f1_3s_local}"
: "${CLEARML_PROJECT:=AI Synthesizer}"

export CLEARML_PROJECT
export CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-f1-3s-gpu4-$(date +%Y%m%d-%H%M)}"

# Runtime dataset config (pre-encoded latents on local disk).
RUNTIME_DATASET_CONFIG="${RUNTIME_DATASET_CONFIG:-${TMPDIR}/f1_dataset_pre_encoded_runtime.json}"
python3 - <<PY
import json, os
from pathlib import Path
path = Path("${PRE_ENCODED_PATH}").resolve()
out = {
    "dataset_type": "pre_encoded",
    "random_crop": False,
    "datasets": [{"id": "f1_pre_encoded_local", "path": str(path)}],
}
Path("${RUNTIME_DATASET_CONFIG}").write_text(json.dumps(out, indent=2) + "\n")
print("Wrote dataset config:", "${RUNTIME_DATASET_CONFIG}", "→", path)
PY

mkdir -p "${SAVE_DIR}"

run_train() {
  "$@" train.py \
    --model-config "${MODEL_CONFIG}" \
    --dataset-config "${RUNTIME_DATASET_CONFIG}" \
    --pretrained-ckpt-path "${PRETRAINED_CKPT_PATH}" \
    --name "${CLEARML_TASK_NAME}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --num-gpus 1 \
    --precision "${PRECISION:-16-mixed}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-5000}" \
    --save-dir "${SAVE_DIR}" \
    --num-workers "${NUM_WORKERS:-16}" \
    --accum-batches "${ACCUM_BATCHES:-1}" \
    --seed "${SEED:-42}"
}

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_DISPLAY="${REPO_ROOT}/.venv/bin/python"
  RUNNER=("${REPO_ROOT}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_DISPLAY="uv run python"
  RUNNER=(uv run python)
else
  PYTHON_DISPLAY="python3"
  RUNNER=(python3)
fi

echo "=== F1 training (GPU ${CUDA_VISIBLE_DEVICES}) ==="
echo "Repo          : ${REPO_ROOT}"
echo "Python        : ${PYTHON_DISPLAY}"
echo "Pre-encoded   : ${PRE_ENCODED_PATH}"
echo "Checkpoint in : ${PRETRAINED_CKPT_PATH}"
echo "Save dir      : ${SAVE_DIR}"
echo "ClearML task  : ${CLEARML_TASK_NAME}"
echo ""

run_train "${RUNNER[@]}"

echo "=== Training finished ==="
