#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
HAB_MODE="${1:?probe or run}"
HAB_LOG="logs/layout100_${HAB_MODE}_v1.log"
exec > >(tee -a "$HAB_LOG") 2>&1
echo "START $(date -Is) mode=$HAB_MODE"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 1 > "logs/layout100_${HAB_MODE}_gpu.csv" &
HAB_MONITOR_PID=$!
trap 'kill "$HAB_MONITOR_PID" 2>/dev/null || true' EXIT
if [ "$HAB_MODE" = probe ]; then
  "$HAB_PYTHON" -u scripts/evaluate_layouts.py --mode probe --output results/validation_layout100_v1
elif [ "$HAB_MODE" = probe2 ]; then
  "$HAB_PYTHON" -u scripts/evaluate_layouts.py --mode probe --probe-repeats 2 --output results/validation_layout100_repair_v1
elif [ "$HAB_MODE" = resume ]; then
  test -f results/validation_layout100_repair_v1/complete_probe_0.json
  test -d results/pick_val_layout100_v1
  for HAB_SLOT in 0 1 2 3 4; do
    "$HAB_PYTHON" -u scripts/evaluate_layouts.py --mode run --resume --layout-slot "$HAB_SLOT" --output results/pick_val_layout100_v1
  done
elif [ "$HAB_MODE" = run ]; then
  test -f results/validation_layout100_v1/complete_probe_0.json
  test ! -e results/pick_val_layout100_v1
  for HAB_SLOT in 0 1 2 3 4; do
    "$HAB_PYTHON" -u scripts/evaluate_layouts.py --mode run --layout-slot "$HAB_SLOT" --output results/pick_val_layout100_v1
  done
else
  exit 2
fi
echo "DONE $(date -Is) mode=$HAB_MODE"
