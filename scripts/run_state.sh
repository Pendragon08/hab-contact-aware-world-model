#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
mode="${1:?witness or pilot}"
case "$mode" in witness|pilot) ;; *)exit 2;; esac
python -m unittest discover -s tests -v
if [ "$mode" = pilot ]; then
  test -f artifacts/state_tdmpc2_witness_v1/summary.json
  test -f artifacts/state_sac_witness_v1/summary.json
fi
for algorithm in tdmpc2 sac; do
  python scripts/train_state_pilot.py --algo "$algorithm" --mode "$mode" 2>&1 | tee "logs/state_${algorithm}_${mode}.log"
done
