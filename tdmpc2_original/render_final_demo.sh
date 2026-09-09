#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
exec "$HAB_PYTHON" "$ORIG_ROOT/render_final_demo.py"
