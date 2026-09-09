#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/local-settings.sh"
source "$ROOT/scripts/runtime.sh"
export PYTHONPATH="$ROOT/tdmpc2_original/upstream/tdmpc2:$PYTHONPATH"
exec "$HAB_PYTHON" "$ROOT/hab_tdmpc2_original/kernel_probe.py"
