#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
exec "$HAB_PYTHON" "$DMC_ROOT/runner.py" "$@"
