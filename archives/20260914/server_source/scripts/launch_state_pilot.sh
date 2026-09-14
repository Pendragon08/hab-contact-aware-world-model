#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
test -f artifacts/state_tdmpc2_witness_v1/summary.json
test -f artifacts/state_sac_witness_v1/summary.json
test ! -d artifacts/state_tdmpc2_pilot_v1
test ! -d artifacts/state_sac_pilot_v1
python scripts/monitor_resources.py --pid $$ --out logs/state_pilot_resources.jsonl &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
bash scripts/run_state.sh pilot
python scripts/report_state_pilot.py 2>&1 | tee logs/state_report.log
date -u +%FT%TZ > artifacts/STATE_PILOTS_FINISHED
