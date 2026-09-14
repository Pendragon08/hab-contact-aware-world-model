#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p release
deadline=$((SECONDS+10800))
while [ ! -f artifacts/contact_v1/FINISHED ] || [ ! -f artifacts/STATE_PILOTS_FINISHED ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo 'Finalization wait timed out; do not claim completion.' > release/FINALIZATION_FAILED
    exit 1
  fi
  if ! screen -ls | grep -Eq 'hab_contact300_v1|hab_state_pilot_v1'; then
    echo 'Experiment sessions ended without both completion markers.' > release/FINALIZATION_FAILED
    exit 1
  fi
  sleep 30
done
if bash scripts/finalize_experiment_data.sh > release/finalization.log 2>&1; then
  echo 'All experiments and audits archived.' > release/FINALIZATION_COMPLETE
else
  echo 'Finalization failed; inspect release/finalization.log and retain all results.' > release/FINALIZATION_FAILED
  exit 1
fi
