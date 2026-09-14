#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
tag="$1"; role="$2"; shift 2
case "$tag/$role" in
  dmc/old) source "$ROOT/../dmc_tdmpc2_original/runtime.sh" ;;
  ms2/old) source "$ROOT/../maniskill2_tdmpc2_original/runtime.sh" ;;
  dmc/new) source "$ROOT/walker_run/runtime.sh" ;;
  ms2/new) source "$ROOT/lift_cube/runtime.sh" ;;
  *) exit 22 ;;
esac
exec "$HAB_PYTHON" "$ROOT/shared_entry.py" "$tag" "$role" "$@"
