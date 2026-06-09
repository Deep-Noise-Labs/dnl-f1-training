#!/usr/bin/env bash
# Register / update a ClearML task that points at train_f1_gpu4.sh (clone in UI or enqueue).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PROJECT="${CLEARML_PROJECT:-AI Synthesizer}"
TASK_NAME="${CLEARML_TEMPLATE_NAME:-F1 3s finetune (GPU4 template)}"
SCRIPT="${REPO_ROOT}/scripts/clearml/train_f1_gpu4.sh"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
else
  PYTHON="uv run python"
fi

chmod +x "${SCRIPT}"

export REPO_ROOT PROJECT TASK_NAME SCRIPT
REPO_ROOT="${REPO_ROOT}"
PROJECT="${PROJECT}"
TASK_NAME="${TASK_NAME}"
SCRIPT="${SCRIPT}"

"${PYTHON}" - <<'PY'
import os
from pathlib import Path

from clearml import Task

repo = Path(os.environ["REPO_ROOT"])
project = os.environ["PROJECT"]
name = os.environ["TASK_NAME"]
script = Path(os.environ["SCRIPT"])

task = Task.init(
    project_name=project,
    task_name=name,
    task_type=Task.TaskTypes.training,
    tags=["foundation-1", "3s", "template", "gpu4", "local-server"],
    reuse_last_task_id=False,
)
task.set_script(
    repository=str(repo),
    branch=None,
    commit=None,
    entry_point=str(script.relative_to(repo)),
    working_dir=str(repo),
)
task.set_parameter("CUDA_VISIBLE_DEVICES", "4", description="Physical GPU index")
task.set_parameter("USE_WANDB", "0", description="ClearML-only logging")
task.set_parameter("BATCH_SIZE", "128", description="Per-GPU batch (tune for VRAM)")
task.set_parameter("PRE_ENCODED_PATH", "", description="Local pre-encoded latent root")
task.set_parameter("PRETRAINED_CKPT_PATH", "", description="stable-audio-open-1.0.ckpt path")
task.connect(
    {"script": str(script), "env_example": str(repo / "scripts/clearml/train_task.env.example")},
    name="Launch Files",
)
print(f"Template task id: {task.id}")
print(f"Open in UI → clone or Enqueue to queue '{os.environ.get('CLEARML_QUEUE', 'gpu4-h100-queue')}'")
task.close()
PY
