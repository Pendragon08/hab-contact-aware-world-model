#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used >= 500 )); then echo "GPU1 occupied: ${used} MiB; no job started"; exit 20; fi
exec "$HAB_PYTHON" "$ORIG_ROOT/runner.py" "$@"
