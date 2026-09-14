#!/usr/bin/env bash
set -euo pipefail
SIX_ROOT="$(cd "$(dirname "$0")" && pwd)"
task="$1"; shift
job=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["root"])' "$SIX_ROOT/registry.json" "$task")
source "$job/runtime.sh"
exec "$HAB_PYTHON" "$SIX_ROOT/entry.py" "$task" "$@"
