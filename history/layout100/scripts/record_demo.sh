#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv -l 1 > logs/demo_v2_gpu.csv &
HAB_MONITOR_PID=$!
trap 'kill "$HAB_MONITOR_PID" 2>/dev/null || true' EXIT
bash scripts/run_demo.sh results/pick_val_seed0_4_v2
