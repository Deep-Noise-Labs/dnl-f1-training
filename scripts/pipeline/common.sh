#!/usr/bin/env bash
# Shared paths for F1 pipeline tmux steps.
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/data/repos/dnl-f1-training}"
export LOG_DIR="${LOG_DIR:-/data/logs/f1-pipeline}"
export MARKER_DIR="${MARKER_DIR:-${REPO_ROOT}/.pipeline}"
export VENV="${REPO_ROOT}/.venv"

mkdir -p "${LOG_DIR}" "${MARKER_DIR}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG_DIR}/pipeline.log"; }

wait_marker() {
  local name="$1"
  local path="${MARKER_DIR}/${name}"
  log "Waiting for ${path} ..."
  while [[ ! -f "${path}" ]]; do
    sleep 30
  done
  log "Found ${path}"
}

mark_done() {
  local name="$1"
  date -Iseconds > "${MARKER_DIR}/${name}"
  log "Marked ${name}"
}

activate_venv() {
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
  cd "${REPO_ROOT}"
}
