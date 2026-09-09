#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
test ! -e "$ORIG_ROOT/artifacts/seed1_500k"
test -f "$ORIG_ROOT/reports/agent-follows-doc.json"
"$HAB_PYTHON" - <<'PY'
import os,json,pathlib,shutil
p=pathlib.Path(os.environ['ORIG_ROOT'])
assert json.loads((p/'reports/agent-follows-doc.json').read_text())['passed'] is True
assert shutil.disk_usage(p).free>15*2**30, 'Less than 15GiB disk reserve'
PY
used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used >= 500 )); then echo "GPU1 occupied: ${used} MiB"; exit 20; fi
screen -dmS tdmpc2_original_mw_seed1 bash -c "bash '$ORIG_ROOT/run.sh' train > '$ORIG_ROOT/logs/train.log' 2>&1 && bash '$ORIG_ROOT/finalize.sh' > '$ORIG_ROOT/logs/finalize.log' 2>&1"
echo ORIGINAL_SCREEN_DISPATCHED_VERIFY_ACTUAL_UPDATES
