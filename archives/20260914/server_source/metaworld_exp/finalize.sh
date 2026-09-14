#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
"$HAB_PYTHON" "$MW_ROOT/report.py"
"$HAB_PYTHON" "$HAB_ROOT/scripts/archive_artifacts.py" --name metaworld-pick-place-v1 --paths metaworld_exp/artifacts metaworld_exp/reports metaworld_exp/logs --part-mib 500
