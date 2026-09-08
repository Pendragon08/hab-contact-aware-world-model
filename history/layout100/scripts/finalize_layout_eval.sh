#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
exec > >(tee logs/layout100_finalize_v1.log) 2>&1
for HAB_SLOT in 0 1 2 3 4; do
  test -f "results/pick_val_layout100_v1/complete_run_${HAB_SLOT}.json"
done
"$HAB_PYTHON" scripts/audit_preserved.py --verify
"$HAB_PYTHON" scripts/summarize_layouts.py --root "$HAB_ROOT"
"$HAB_PYTHON" scripts/validate_layout_media.py --root "$HAB_ROOT"
"$HAB_PYTHON" scripts/build_layout_report.py
sha256sum assets/data/mshab_checkpoints/rl/tidy_house/pick/all/policy.pt
echo "FINALIZATION_DONE $(date -Is)"
