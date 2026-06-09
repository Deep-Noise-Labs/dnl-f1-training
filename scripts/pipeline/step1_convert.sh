#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step1.log" 2>&1

log "=== Step 1: convert instrs → F1 sidecars ==="
if [[ -x "${VENV}/bin/python" ]]; then
  activate_venv
  PY=python
else
  cd "${REPO_ROOT}"
  PY=python3
fi

IN="/data/aisynth_datasets/training_datasets/instrs_C3_midasheng"
OUT="/data/aisynth_datasets/training_datasets/f1_instrs_C3_midasheng"

need=0
for s in train valid test; do
  src=$(find "${IN}/${s}" -name '*.json' | wc -l)
  dst=$(find "${OUT}/${s}" -name '*.json' 2>/dev/null | wc -l)
  log "${s}: src=${src} out=${dst}"
  if [[ "${src}" != "${dst}" ]]; then need=1; fi
done

if [[ "${need}" -eq 0 ]]; then
  log "Conversion already complete — skipping"
else
  "${PY}" scripts/convert_instrs_to_f1.py --input "${IN}" --output "${OUT}" --symlink-wav
fi
mark_done step1.done
