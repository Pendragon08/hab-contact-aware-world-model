#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
stage="${1:?witness, interface, collect or train}"
case "$stage" in witness|interface|collect|train) ;; *) exit 2;; esac
version=$(python -c 'import os,json; print(json.load(open(os.environ.get("HAB_BC_CONFIG","configs/bc_v1.json")))["version"])')
test ! -f "logs/${version}_${stage}.log"
python scripts/monitor_resources.py --pid $$ --out "logs/${version}_${stage}_resources.jsonl" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/bc_pipeline.py "$stage" 2>&1 | tee "logs/${version}_${stage}.log"
