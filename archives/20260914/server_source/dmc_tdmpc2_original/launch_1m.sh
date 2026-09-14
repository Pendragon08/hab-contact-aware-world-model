#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
exec 9>"$DMC_ROOT/launch_1m.lock"
flock -n 9 || { echo '1M launch already in progress'; exit 21; }
test ! -e "$DMC_ROOT/launch_1m.json" || { echo '1M already submitted'; exit 23; }
"$HAB_PYTHON" "$DMC_ROOT/check_budget.py"
"$HAB_PYTHON" - <<'PY'
import runner,json
assert json.loads((runner.ROOT/'reports/independent-budget-1m.json').read_text())['passed'] is True
print(runner.resource_guard(require_idle=True))
PY
screen -dmS tdmpc2_dmc_walker_3seed_1m_gpu0 bash -c 'bash "$1/queue_1m.sh" > "$1/logs/queue_1m.log" 2>&1' _ "$DMC_ROOT"
printf '{"submitted":true,"screen":"tdmpc2_dmc_walker_3seed_1m_gpu0","gpu":0,"seeds":[1,2,3],"environment_steps_per_seed":1000000}\n' > "$DMC_ROOT/launch_1m.json"
screen -ls
