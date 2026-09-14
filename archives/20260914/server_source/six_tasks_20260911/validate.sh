#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/validation"
(set -o noclobber; date -u > "$ROOT/validation/submitted")
# Leave the uninterrupted four-task baseline intact before adding witnesses.
while ! python3 - "$ROOT/measurements/four_baseline.jsonl" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1])]
sys.exit(0 if rows[-1]['time']-rows[0]['time']>=300 else 1)
PY
do sleep 5; done
set +e
bash "$ROOT/run.sh" cheetah-run witness --seed 1 > "$ROOT/validation/cheetah-run.log" 2>&1 &
dmc=$!
bash "$ROOT/run.sh" stack-cube witness --seed 1 > "$ROOT/validation/stack-cube.log" 2>&1 &
ms2=$!
wait "$dmc"; drc=$?
wait "$ms2"; mrc=$?
echo "WITNESS_EXIT_CODES cheetah-run=$drc stack-cube=$mrc"
if [[ "$drc" -ne 0 || "$mrc" -ne 0 ]]; then
  tail -n 35 "$ROOT/validation/cheetah-run.log" "$ROOT/validation/stack-cube.log"
  exit 1
fi
echo SIX_TASK_WITNESSES_PASS
