#!/usr/bin/env bash
# Create / refresh the project .venv for local GPU training (pip, staged installs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON_BIN="python${PYTHON_VERSION}"
VENV_DIR="${REPO_ROOT}/.venv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv python install "${PYTHON_VERSION}" >/dev/null 2>&1 || true
  fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found (try: uv python install ${PYTHON_VERSION})" >&2
  exit 1
fi

echo "=== Training venv at ${VENV_DIR} (Python ${PYTHON_VERSION}) ==="
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel 'setuptools>=68,<81'
pip install 'numpy<2' 'protobuf>=3.20,<4.21'

if ! python -c "import torch" 2>/dev/null; then
  echo "=== Installing PyTorch 2.2 (CUDA 12.1) ==="
  pip install --upgrade \
    'torch==2.2.2+cu121' \
    'torchaudio==2.2.2+cu121' \
    --index-url https://download.pytorch.org/whl/cu121
fi

# Staged installs avoid pip "resolution-too-deep" (wandb vs google-cloud-storage/protobuf).
echo "=== Installing training stack (staged) ==="
pip install \
  'pytorch-lightning>=2.0,<2.2' \
  'torchmetrics>=1.0,<2.0' \
  'wandb>=0.15.4,<0.16' \
  'clearml>=1.14' \
  'huggingface_hub>=0.20' \
  'safetensors>=0.4' \
  'einops>=0.7,<0.8' \
  'einops-exts>=0.0.4,<0.1' \
  'ema-pytorch>=0.2.3,<0.3' \
  'scipy>=1.8.1,<1.14' \
  'tqdm' \
  'prefigure>=0.0.9,<0.1' \
  'transformers==4.35.2' \
  'sentencepiece>=0.1.99,<0.2' \
  'laion-clap>=1.1.4,<1.2' \
  'auraloss>=0.4,<0.5' \
  'aeiou>=0.0.20,<0.1' \
  'k-diffusion>=0.1.1,<0.2' \
  'v-diffusion-pytorch>=0.0.2,<0.1' \
  'x-transformers>=1.26,<1.27' \
  'local-attention>=1.8.6,<1.9' \
  'vector-quantize-pytorch>=1.9.14,<1.10' \
  'descript-audio-codec>=1.0,<1.1' \
  'encodec>=0.1.1,<0.2' \
  'alias-free-torch>=0.0.6,<0.1' \
  'PyWavelets>=1.4.1,<1.5' \
  'soundfile>=0.12.1' \
  'pydub>=0.25.1' \
  'librosa>=0.10,<0.11' \
  'soxr>=0.3.7' \
  'pandas>=2.0,<3' \
  'matplotlib>=3.7,<3.8' \
  'importlib-resources>=5.12,<6' \
  'pretty_midi>=0.2.9,<0.3' \
  'webdataset>=0.2.48,<0.3' \
  's3fs>=2023.10' \
  'google-cloud-storage>=2.14,<3' \
  'deepspeed>=0.12' \
  'bitsandbytes>=0.41' \
  clearml-agent

pip install -e . --no-deps

# Keep torchaudio on 2.2.x so WAV loading uses soundfile (not torchcodec).
echo "=== Pinning torch/torchaudio 2.2.2+cu121 ==="
pip install 'numpy<2' --force-reinstall
pip install --force-reinstall \
  'torch==2.2.2+cu121' \
  'torchaudio==2.2.2+cu121' \
  'torchvision==0.17.2+cu121' \
  --index-url https://download.pytorch.org/whl/cu121
pip install 'transformers==4.35.2' --force-reinstall

echo "=== Verify imports ==="
python - <<'PY'
import torch
import pytorch_lightning as pl
import stable_audio_tools
import clearml
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("pytorch_lightning", pl.__version__)
print("clearml", clearml.__version__)
PY

echo "=== Done. Activate with: source ${VENV_DIR}/bin/activate ==="
