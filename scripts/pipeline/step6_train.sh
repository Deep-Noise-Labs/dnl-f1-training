#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step6.log" 2>&1

wait_marker step4.done
wait_marker step5.done

log "=== Step 6: DiT fine-tune on GPU 4 ==="
activate_venv
export CLEARML_TRAIN_ENV="${REPO_ROOT}/scripts/clearml/train_task.env"
# shellcheck source=/dev/null
source "${CLEARML_TRAIN_ENV}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export USE_WANDB=0
export CLEARML_PROJECT
export CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-f1-instrs-gpu4-$(date +%Y%m%d-%H%M)}"

RUNTIME_DATASET_CONFIG="/tmp/f1_dataset_pre_encoded_runtime.json"
RUNTIME_MODEL_CONFIG="/tmp/model_config_train_runtime.json"

python - <<PY
import json
from pathlib import Path
Path("${RUNTIME_DATASET_CONFIG}").write_text(json.dumps({
    "dataset_type": "pre_encoded",
    "random_crop": False,
    "datasets": [{"id": "f1_pre_encoded_local", "path": str(Path("${PRE_ENCODED_PATH}").resolve())}],
}, indent=2) + "\n")
cfg = json.loads(Path("${MODEL_CONFIG}").read_text())
for c in cfg.get("model", {}).get("conditioning", {}).get("configs", []):
    if c.get("type") in ("clap_text", "clap_audio"):
        c.setdefault("config", {})["clap_ckpt_path"] = "${CLAP_CKPT_PATH}"
Path("${RUNTIME_MODEL_CONFIG}").write_text(json.dumps(cfg, indent=2) + "\n")
print("dataset + model runtime configs written")
PY

mkdir -p "${SAVE_DIR}"

python train.py \
  --model-config "${RUNTIME_MODEL_CONFIG}" \
  --dataset-config "${RUNTIME_DATASET_CONFIG}" \
  --pretrained-ckpt-path "${PRETRAINED_CKPT_PATH}" \
  --name "${CLEARML_TASK_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-gpus 1 \
  --precision "${PRECISION}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --save-dir "${SAVE_DIR}" \
  --num-workers "${NUM_WORKERS}" \
  --accum-batches "${ACCUM_BATCHES}" \
  --seed "${SEED}"

mark_done step6.done
