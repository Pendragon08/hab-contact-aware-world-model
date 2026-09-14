#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
test -f artifacts/contact_witness_v1/complete.json
test ! -d artifacts/contact_v1/sac
python scripts/monitor_resources.py --pid $$ --out logs/contact_formal_resources.jsonl &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
bash scripts/run_contact.sh run all
python scripts/report_contact.py 2>&1 | tee logs/contact_report.log
date -u +%FT%TZ > artifacts/contact_v1/FINISHED
