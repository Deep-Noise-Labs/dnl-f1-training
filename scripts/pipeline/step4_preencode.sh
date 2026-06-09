#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step4.log" 2>&1

wait_marker step0.done
wait_marker step1.done
wait_marker step2.done
wait_marker step3.done

export CUDA_VISIBLE_DEVICES="${PREENCODE_GPU:-7}"
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export CLEARML_PROJECT="${CLEARML_PROJECT:-AI Synthesizer}"
export CLEARML_TASK_NAME="${CLEARML_TASK_NAME:-f1-preencode-instrs-$(date +%Y%m%d-%H%M)}"
log "=== Step 4: VAE pre-encode on GPU ${CUDA_VISIBLE_DEVICES} ==="
activate_venv

OUT="/data/aisynth_datasets/pre_encoded/f1_instrs_C3_midasheng"
CONFIG="/data/aisynth_datasets/configs/f1_instrs_pre_encode.json"
CKPT="/data/checkpoints/stable-audio-open-1.0.ckpt"
CLAP="/data/checkpoints/clap.ckpt"
MODEL_RUNTIME="/tmp/model_config_preencode_runtime.json"

mkdir -p "${OUT}"

python - <<PY
import json
from pathlib import Path
src = Path("${REPO_ROOT}/models/foundation1_3s/model_config_3s.json")
cfg = json.loads(src.read_text())
for c in cfg.get("model", {}).get("conditioning", {}).get("configs", []):
    if c.get("type") in ("clap_text", "clap_audio"):
        c.setdefault("config", {})["clap_ckpt_path"] = "${CLAP}"
Path("${MODEL_RUNTIME}").write_text(json.dumps(cfg, indent=2))
print("Wrote", "${MODEL_RUNTIME}")
PY

python pre_encode.py \
  --model-config "${MODEL_RUNTIME}" \
  --ckpt-path "${CKPT}" \
  --dataset-config "${CONFIG}" \
  --output-path "${OUT}" \
  --sample-size 131072 \
  --expected-sample-size 131072 \
  --batch-size 32 \
  --num-workers 16

mark_done step4.done
