#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
export HAB_BC_CONFIG=configs/bc_v1_2.json
mode="${1:?witness or run}"
test ! -f "logs/bc_v1_2_eval_v2_${mode}.log"
python scripts/monitor_resources.py --pid $$ --out "logs/bc_v1_2_eval_v2_${mode}_resources.jsonl" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/bc_eval_v2.py "$mode" 2>&1 | tee "logs/bc_v1_2_eval_v2_${mode}.log"
