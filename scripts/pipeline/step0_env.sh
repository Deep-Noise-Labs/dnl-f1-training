#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step0.log" 2>&1

log "=== Step 0: uv training environment ==="
cd "${REPO_ROOT}"
bash scripts/clearml/install_training_env.sh
activate_venv
python -c "import torch, clearml, stable_audio_tools; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mark_done step0.done
