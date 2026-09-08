#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
HAB_OUTPUT="${1:-results/pick_val_seed0_4}"
if [ -d "$HAB_OUTPUT" ]; then
  echo "Output directory already exists. Pass a new relative result path to preserve earlier runs."
  exit 2
fi
"$HAB_PYTHON" -u scripts/demo.py --episodes 5 --start-seed 0 --output "$HAB_OUTPUT"
