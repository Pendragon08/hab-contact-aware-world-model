#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec 9>"$ROOT/launch.lock"
flock -n 9
test -f "$ROOT/independent-validation.json"
test ! -e "$ROOT/launch.json"
test ! -e "$ROOT/handoff.json"
python3 - "$ROOT" <<'PY'
import json,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);r=json.loads((p/'independent-validation.json').read_text());assert r['passed']
(p/'launch.json').write_text(json.dumps(dict(utc_epoch=time.time(),screen='tdmpc2_six_tasks_1m'))+'\n')
PY
screen -dmS tdmpc2_six_tasks_1m bash -c 'exec python3 "$1/manager.py" >> "$1/manager.log" 2>&1' _ "$ROOT"
echo SIX_TASK_MANAGER_SUBMITTED
