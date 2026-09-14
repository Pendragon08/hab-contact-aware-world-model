#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
exec 9>"$DMC_ROOT/launch.lock"
flock -n 9 || { echo 'Launch already in progress'; exit 21; }
test -f "$DMC_ROOT/reports/independent-witness.json" || { echo 'Missing independently checked witness'; exit 22; }
test ! -e "$DMC_ROOT/launch.json" || { echo 'Already submitted; not launching again'; exit 23; }
"$HAB_PYTHON" - <<'PY'
import json,pathlib,subprocess,shutil,os
p=pathlib.Path(os.environ['DMC_ROOT'])
report=json.loads((p/'reports/independent-witness.json').read_text())
assert report['passed'] is True
assert json.loads(pathlib.Path(report['remote_complete']).read_text())['completed'] is True
used,free=map(int,subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],text=True).split(','))
assert used<500 and free>=12288 and shutil.disk_usage(p).free>=15*2**30
PY
screen -dmS tdmpc2_dmc_walker_3seed_gpu0 bash -c 'bash "$1/queue.sh" > "$1/logs/queue.log" 2>&1' _ "$DMC_ROOT"
printf '{"submitted":true,"screen":"tdmpc2_dmc_walker_3seed_gpu0","gpu":0,"seeds":[1,2,3]}\n' > "$DMC_ROOT/launch.json"
screen -ls
