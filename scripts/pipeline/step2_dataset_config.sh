#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step2.log" 2>&1

log "=== Step 2: pre-encode dataset config ==="
DATA_ROOT="/data/aisynth_datasets/training_datasets/f1_instrs_C3_midasheng"
CONFIG="/data/aisynth_datasets/configs/f1_instrs_pre_encode.json"
mkdir -p "$(dirname "${CONFIG}")"

cat > "${CONFIG}" <<EOF
{
  "dataset_type": "audio_dir",
  "random_crop": false,
  "datasets": [
    {
      "id": "f1_train",
      "path": "${DATA_ROOT}/train"
    },
    {
      "id": "f1_valid",
      "path": "${DATA_ROOT}/valid"
    }
  ]
}
EOF
log "Wrote ${CONFIG}"
mark_done step2.done
