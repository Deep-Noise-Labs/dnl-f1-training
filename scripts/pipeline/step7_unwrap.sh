#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step7.log" 2>&1

wait_marker step6.done

log "=== Step 7: unwrap latest checkpoint ==="
SAVE_DIR="/data/checkpoints/f1_3s_local"
OUT_DIR="${REPO_ROOT}/models/foundation1_3s"

CKPT=$(find "${SAVE_DIR}" -name '*.ckpt' -type f 2>/dev/null | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
  log "ERROR: no .ckpt under ${SAVE_DIR}" >&2
  exit 1
fi
log "Unwrapping ${CKPT}"
STEP=$(basename "${CKPT}" .ckpt | sed -n 's/.*step=\([0-9]*\).*/\1/p')
STEP="${STEP:-unknown}"
bash "${REPO_ROOT}/scripts/unwrap_checkpoint.sh" "${CKPT}" "f1_instrs_C3_step${STEP}"
mark_done step7.done
log "=== Pipeline complete ==="
