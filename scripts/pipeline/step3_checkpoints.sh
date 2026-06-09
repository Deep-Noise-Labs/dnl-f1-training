#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec >> "${LOG_DIR}/step3.log" 2>&1

wait_marker step0.done
log "=== Step 3: download / link checkpoints ==="
activate_venv

CKPT_DIR="/data/checkpoints"
mkdir -p "${CKPT_DIR}"

STABLE="${CKPT_DIR}/stable-audio-open-1.0.ckpt"
CLAP="${CKPT_DIR}/clap.ckpt"
FOUNDATION="${CKPT_DIR}/Foundation_1.safetensors"

if [[ ! -f "${STABLE}" ]]; then
  log "Downloading stabilityai/stable-audio-open-1.0 model.safetensors ..."
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
    log "ERROR: set HF_TOKEN or HUGGINGFACE_TOKEN for gated stable-audio-open-1.0" >&2
    exit 1
  fi
  export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
dst = Path("/data/checkpoints/stable-audio-open-1.0.ckpt")
path = hf_hub_download(
    repo_id="stabilityai/stable-audio-open-1.0",
    filename="model.safetensors",
    local_dir="/data/checkpoints/hf_stable_audio",
    token=token,
)
shutil.copy2(path, dst)
print("Saved", dst, "size", dst.stat().st_size)
PY
fi

if [[ ! -f "${FOUNDATION}" ]]; then
  log "Downloading RoyalCities/Foundation-1 ..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil
dst = Path("/data/checkpoints/Foundation_1.safetensors")
path = hf_hub_download(
    repo_id="RoyalCities/Foundation-1",
    filename="Foundation_1.safetensors",
    local_dir="/data/checkpoints/hf_foundation1",
)
shutil.copy2(path, dst)
print("Saved", dst)
PY
fi

if [[ ! -f "${CLAP}" ]]; then
  CLAP_SRC="/data/repos/audiocraft/metric_models/clap/music_audioset_epoch_15_esc_90.14.pt"
  if [[ -f "${CLAP_SRC}" ]]; then
    cp -a "${CLAP_SRC}" "${CLAP}"
    log "Copied CLAP from ${CLAP_SRC}"
  else
    log "ERROR: CLAP not found at ${CLAP_SRC}" >&2
    exit 1
  fi
fi

for f in "${STABLE}" "${CLAP}"; do
  [[ -f "${f}" ]] || { log "Missing ${f}"; exit 1; }
  log "OK ${f} ($(du -h "${f}" | cut -f1))"
done
mark_done step3.done
