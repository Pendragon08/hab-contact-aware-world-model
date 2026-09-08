#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
test ! -e logs/bc_v1_2_sequence.log || exit 2
bash scripts/run_bc_sequence.sh > logs/bc_v1_2_sequence.log 2>&1
status=$?
printf '{"exit_code":%d}\n' "$status" > logs/bc_v1_2_sequence_exit.json
exit "$status"
