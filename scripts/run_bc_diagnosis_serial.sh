#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
export HAB_BC_CONFIG=configs/bc_v1_2.json
stage="${1:?witness or run}"
test ! -f "logs/bc_diagnosis_v1_serial_${stage}.log"
python - <<'PY'
import subprocess
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
g={int(a):[int(b),int(c)] for a,b,c in (line.split(',') for line in rows.splitlines())}
assert g[0][1]>=12288 and g[1][0]<500,g
print('SERIAL_RESOURCE_GUARD_PASS',g,flush=True)
PY
python scripts/monitor_resources.py --pid $$ --out "logs/bc_diagnosis_v1_serial_${stage}_resources.jsonl" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
python scripts/bc_diagnosis_serial.py "$stage" 2>&1 | tee "logs/bc_diagnosis_v1_serial_${stage}.log"
