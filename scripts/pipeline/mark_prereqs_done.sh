#!/usr/bin/env bash
# Mark steps 1–3 complete when data conversion and checkpoints are ready.
set -euo pipefail
source "$(dirname "$0")/common.sh"
mark_done step1.done
mark_done step2.done
mark_done step3.done
log "Marked step1, step2, step3 done"
