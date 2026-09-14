#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec 9>"$ROOT/launch.lock"
flock -n 9
test ! -e "$ROOT/submitted.json"
test ! -e "$ROOT/started.json"
screen -dmS tdmpc2_throughput_2pergpu bash -c 'exec python3 "$1/controller.py" > "$1/controller.log" 2>&1' _ "$ROOT"
printf '{"submitted":true,"screen":"tdmpc2_throughput_2pergpu"}\n' > "$ROOT/submitted.json"
screen -ls
