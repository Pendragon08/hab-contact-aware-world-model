#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
export HAB_BC_CONFIG=configs/bc_v1_2.json
mode="${1:?witness or run}"
case "$mode" in witness|run) ;; *) exit 2;; esac
test ! -e "artifacts/demo_tdmpc2_v1/$mode"
test ! -e "logs/demo_tdmpc2_v1_${mode}.log"
python scripts/monitor_resources.py --pid $$ --out "logs/demo_tdmpc2_v1_${mode}_resources.jsonl" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/train_demo_tdmpc2.py "$mode" 2>&1 | tee "logs/demo_tdmpc2_v1_${mode}.log"
