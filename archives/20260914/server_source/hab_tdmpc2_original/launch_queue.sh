#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/local-settings.sh"
test ! -e "$ROOT/hab_tdmpc2_original/artifacts/queue"
test ! -e "$ROOT/hab_tdmpc2_original/artifacts/seed1_500k"
mkdir -p "$ROOT/hab_tdmpc2_original/logs"
export HAB_PYTHON
screen -dmS hab_original_after_metaworld bash -c "'$HAB_PYTHON' '$ROOT/hab_tdmpc2_original/wait_dependency.py' > '$ROOT/hab_tdmpc2_original/logs/queue.log' 2>&1"
echo HAB_QUEUE_DISPATCHED_VERIFY_STATUS_FILE
