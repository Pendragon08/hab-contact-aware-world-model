#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
exec 9>"$MS_ROOT/launch.lock"
flock -n 9 || { echo 'Launch already active'; exit 21; }
test ! -e "$MS_ROOT/launch.json" || { echo 'Already submitted'; exit 22; }
"$HAB_PYTHON" - <<'PY'
import runner,json,hashlib,pathlib
report=json.loads((runner.ROOT/'reports/independent-witness.json').read_text())
assert report['passed'] is True
complete=pathlib.Path(report['remote_complete'])
assert json.loads(complete.read_text())['completed'] is True
for name,digest in json.loads((complete.parent/'sources.json').read_text()).items():
 assert hashlib.sha256((runner.ROOT/name).read_bytes()).hexdigest()==digest,name
print(runner.resource_guard(require_idle=True))
PY
screen -dmS tdmpc2_ms2_pickcube_3seed_1m_gpu1 bash -c 'bash "$1/queue.sh" > "$1/logs/queue.log" 2>&1' _ "$MS_ROOT"
printf '{"submitted":true,"screen":"tdmpc2_ms2_pickcube_3seed_1m_gpu1","gpu":1,"seeds":[1,2,3],"environment_steps_per_seed":1000000}\n' > "$MS_ROOT/launch.json"
screen -ls
