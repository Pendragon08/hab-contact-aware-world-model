#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
for seed in 1 2 3; do
  echo "$(date -Is) starting_seed=$seed"
  "$HAB_PYTHON" "$DMC_ROOT/runner.py" train --seed "$seed" > "$DMC_ROOT/logs/seed${seed}.log" 2>&1
  "$HAB_PYTHON" "$DMC_ROOT/finalize.py" --seed "$seed"
  echo "$(date -Is) completed_seed=$seed"
done
echo DMC_ALL_THREE_SEEDS_COMPLETE
