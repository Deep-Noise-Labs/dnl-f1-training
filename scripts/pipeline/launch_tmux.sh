#!/usr/bin/env bash
# Launch F1 pipeline steps 0–7 in separate tmux sessions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="/data/logs/f1-pipeline"
PIPE="${REPO_ROOT}/scripts/pipeline"

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/.pipeline"
chmod +x "${PIPE}"/*.sh

STEPS=(
  "step0_env.sh"
  "step1_convert.sh"
  "step2_dataset_config.sh"
  "step3_checkpoints.sh"
  "step4_preencode.sh"
  "step5_train_env.sh"
  "step6_train.sh"
  "step7_unwrap.sh"
)

for i in "${!STEPS[@]}"; do
  session="f1-step${i}"
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    "export REPO_ROOT='${REPO_ROOT}'; cd '${REPO_ROOT}'; set +e; bash '${PIPE}/${STEPS[$i]}'; ec=\$?; echo; echo '=== step${i} exited with code '\${ec}' ===; sleep infinity"
  echo "Started tmux session: ${session} → ${STEPS[$i]}"
done

echo ""
echo "Attach:  tmux attach -t f1-step4   (pre-encode)"
echo "List:    tmux ls"
echo "Logs:    ${LOG_DIR}/step*.log"
