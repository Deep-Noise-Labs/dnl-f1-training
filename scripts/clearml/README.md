# ClearML training on this server (uv + GPU 4)

Run Foundation-1 fine-tuning on the local 8×H100 host with **ClearML** as the experiment tracker and **`uv`** for the Python environment. Vertex AI / Docker still use `pip` + `requirements/*.txt`; this path is for **bare-metal** jobs on GPU 4.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed (`uv --version`)
- ClearML credentials configured (`clearml-init` or `~/.clearml.conf`)
- Pre-encoded latents on disk (after `pre_encode.py` or GCS rsync)
- Base checkpoint: `stable-audio-open-1.0.ckpt` (+ CLAP weights if your runtime config needs them)

## 1. Install the training environment (uv)

From the repo root:

```bash
cd /data/repos/dnl-f1-training

# Dev tools only (pytest) — does not install torch/wandb from setup.py
uv sync --only-dev --no-install-project

# Full GPU training stack into .venv/ (Python 3.11 per .python-version)
bash scripts/clearml/install_training_env.sh
source .venv/bin/activate
```

Equivalent manual steps:

```bash
uv venv -p 3.11
source .venv/bin/activate
uv sync --group dev
uv pip install -U pip setuptools wheel
uv pip install -r requirements/base.txt -r requirements/train.txt
uv pip install -e . --no-deps
uv pip install clearml-agent
```

## 2. Configure paths

```bash
cp scripts/clearml/train_task.env.example scripts/clearml/train_task.env
# Edit PRE_ENCODED_PATH, PRETRAINED_CKPT_PATH, BATCH_SIZE, ClearML keys
```

## 3. Convert AISynth data (optional)

```bash
uv run python scripts/convert_instrs_to_f1.py \
  --input /data/aisynth_datasets/training_datasets/instrs_C3_midasheng \
  --output /data/aisynth_datasets/training_datasets/f1_instrs_C3_midasheng \
  --symlink-wav
```

Then pre-encode with `uv run python pre_encode.py` (see `README_DNL.md`).

## 4. Run tests (uv)

```bash
uv sync --only-dev --no-install-project
.venv/bin/pytest tests/test_convert_instrs_to_f1.py -v
.venv/bin/pytest tests/ --ignore=tests/test_docker_build.py -v
```

## 5. ClearML queue on GPU 4 only

Create the queue once (UI or CLI):

```bash
source .venv/bin/activate
clearml-agent create --name gpu4-h100-queue
```

Start the agent (pins **physical GPU 4**, one logical GPU inside the worker):

```bash
bash scripts/clearml/agent_gpu4_daemon.sh
# Or foreground: CUDA_VISIBLE_DEVICES=4 clearml-agent daemon --queue gpu4-h100-queue --gpus 1
```

Verify isolation:

```bash
CUDA_VISIBLE_DEVICES=4 nvidia-smi
```

## 6. Launch training

**Direct (no queue):**

```bash
source .venv/bin/activate
export CLEARML_TRAIN_ENV=scripts/clearml/train_task.env
bash scripts/clearml/train_f1_gpu4.sh
```

**Via ClearML template task:**

```bash
source .venv/bin/activate
export REPO_ROOT=/data/repos/dnl-f1-training
export CLEARML_PROJECT="AI Synthesizer"
bash scripts/clearml/register_train_template.sh
# In ClearML UI: clone task → Enqueue → gpu4-h100-queue
```

## 7. Maximize GPU 4 utilization

| Knob | Starting point (H100 80GB) |
|------|----------------------------|
| `BATCH_SIZE` | 128 → 192 until OOM, then step down |
| `NUM_WORKERS` | 16–32 |
| `PRECISION` | `16-mixed` |
| `CHECKPOINT_EVERY` | 5000 (not 10000 on short runs) |

Watch ClearML scalars **`:monitor:gpu`** and **GPU Memory (GB)**. Target ~70–75 GB allocated and sustained high utilization; if util is low, increase `NUM_WORKERS` or move latents to local SSD.

## 8. Disk layout (avoid root `/` filling up)

| Volume | Size | Use for |
|--------|------|---------|
| `/` (root) | ~193 GB | OS, `/tmp` — **fills quickly** |
| `/data` | ~23 TB | latents, checkpoints, `TMPDIR` |

Each F1 checkpoint is **~15 GB**. Defaults in `train_task.env.example`:

- `SAVE_DIR=/data/checkpoints/...` — keep checkpoints on `/data`
- `CHECKPOINT_SAVE_TOP_K=2` — retain only the last two periodic saves (+ `save_last`)
- `TMPDIR=/data/tmp` — ClearML zip staging and runtime JSON off root

**Do not** upload the whole checkpoint directory to ClearML as an artifact (zips under `/tmp`). The training callback records the local path as a task parameter instead.

If root is full, check `du -sh /tmp/artifacts_*` for stale failed uploads and remove them.

## Files

| File | Role |
|------|------|
| `train_f1_gpu4.sh` | Agent / manual entrypoint |
| `train_task.env.example` | Paths and hyperparameters |
| `install_training_env.sh` | `uv` venv + training deps |
| `agent_gpu4_daemon.sh` | Detached agent on GPU 4 |
| `register_train_template.sh` | Creates a cloneable ClearML task |
