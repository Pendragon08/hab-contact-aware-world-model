#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec 9>"$ROOT/launch.lock"
flock -n 9
test ! -e "$ROOT/launch.json"
test ! -e "$ROOT/handoff.json"
test -e "$ROOT/independent-validation.json"
screen -dmS tdmpc2_four_tasks_1m bash -c 'exec python3 "$1/manager.py" > "$1/manager.log" 2>&1' _ "$ROOT"
printf '{"submitted":true,"screen":"tdmpc2_four_tasks_1m","max_training_processes_per_gpu":2}\n' > "$ROOT/launch.json"
screen -ls
