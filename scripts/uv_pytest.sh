#!/usr/bin/env bash
# Run structural tests with uv-managed dev deps only (no torch/setup.py resolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
uv sync --only-dev --no-install-project
exec "${REPO_ROOT}/.venv/bin/pytest" "$@"
