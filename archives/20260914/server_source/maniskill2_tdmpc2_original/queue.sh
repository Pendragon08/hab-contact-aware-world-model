#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
for seed in 1 2 3; do
  echo "$(date -Is) starting_pick_cube_seed=$seed budget_environment_steps=1000000"
  "$HAB_PYTHON" "$MS_ROOT/runner.py" train --seed "$seed" > "$MS_ROOT/logs/seed${seed}_1m.log" 2>&1
  "$HAB_PYTHON" "$MS_ROOT/finalize.py" --seed "$seed"
  echo "$(date -Is) completed_seed=$seed"
done
echo MS2_ALL_THREE_SEEDS_1M_COMPLETE
