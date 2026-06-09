#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step5.log" 2>&1

log "=== Step 5: train_task.env ==="
ENV_FILE="${REPO_ROOT}/scripts/clearml/train_task.env"
cat > "${ENV_FILE}" <<'EOF'
PRE_ENCODED_PATH=/data/aisynth_datasets/pre_encoded/f1_instrs_C3_midasheng
PRETRAINED_CKPT_PATH=/data/checkpoints/stable-audio-open-1.0.ckpt
CLAP_CKPT_PATH=/data/checkpoints/clap.ckpt
MODEL_CONFIG=/data/repos/dnl-f1-training/models/foundation1_3s/model_config_3s.json
SAVE_DIR=/data/checkpoints/f1_3s_local
BATCH_SIZE=128
NUM_WORKERS=16
PRECISION=16-mixed
CHECKPOINT_EVERY=5000
ACCUM_BATCHES=1
SEED=42
CLEARML_PROJECT="AI Synthesizer"
CUDA_VISIBLE_DEVICES=4
USE_WANDB=0
EOF
log "Wrote ${ENV_FILE}"
mark_done step5.done
