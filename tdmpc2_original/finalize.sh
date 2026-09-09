#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
test -f "$ORIG_ROOT/artifacts/seed1_500k/complete.json"
"$HAB_PYTHON" "$ORIG_ROOT/../scripts/archive_artifacts.py" --name tdmpc2-original-mw-v2-seed1-500k --paths tdmpc2_original/artifacts tdmpc2_original/reports tdmpc2_original/logs --part-mib 500
