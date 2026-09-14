#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/validation"
test ! -e "$ROOT/validation/submitted"
date -Is > "$ROOT/validation/submitted"
bash "$ROOT/shared_run.sh" dmc new witness > "$ROOT/validation/dmc.log" 2>&1 &
dmc_pid=$!
bash "$ROOT/shared_run.sh" ms2 new witness > "$ROOT/validation/ms2.log" 2>&1 &
ms2_pid=$!
dmc_rc=0; ms2_rc=0
wait "$dmc_pid" || dmc_rc=$?
wait "$ms2_pid" || ms2_rc=$?
printf 'WITNESS_EXIT_CODES dmc=%s ms2=%s\n' "$dmc_rc" "$ms2_rc"
test "$dmc_rc" = 0 && test "$ms2_rc" = 0
echo ADDITIONAL_TASK_WITNESSES_PASS
