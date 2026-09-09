#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
mode="${1:?Expected collect, witness or run}"
case "$mode" in
  collect) timeout --signal=TERM 1800 "$HAB_PYTHON" -u "$MW_ROOT/env_data.py" 2>&1 | tee "$MW_ROOT/logs/collect.log" ;;
  witness|run) "$HAB_PYTHON" -u "$MW_ROOT/train.py" "$mode" 2>&1 | tee "$MW_ROOT/logs/$mode.log" ;;
  *) exit 2 ;;
esac
