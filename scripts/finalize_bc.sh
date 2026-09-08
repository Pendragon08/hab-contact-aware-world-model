#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
export HAB_BC_CONFIG=configs/bc_v1_2.json
python - <<'PY'
import json,pathlib
root=pathlib.Path.cwd()
assert json.loads((root/'logs/bc_v1_2_sequence_exit.json').read_text())['exit_code']==0
assert json.loads((root/'reports/bc_v1_2/audit.json').read_text())['passed']
PY
python scripts/audit_bc_sources.py
python scripts/audit_bc_training.py
python scripts/audit_bc_semantics.py
python scripts/report_bc_resources.py
python scripts/write_bc_chinese.py
python scripts/write_bc_semantics.py
python - <<'PY'
import pathlib,shutil,json,subprocess
root=pathlib.Path.cwd(); target=root/'artifacts/bc_v1_2/finalizer_source';target.mkdir(exist_ok=False)
for name in ['audit_bc_sources.py','audit_bc_training.py','audit_bc_semantics.py','report_bc_resources.py','write_bc_chinese.py','write_bc_semantics.py','finalize_bc.sh']:
    shutil.copyfile(root/'scripts'/name,target/name)
result=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
(root/'reports/bc_v1_2/gpu_after.txt').write_text(result)
PY
python scripts/archive_artifacts.py --name bc-pilot-v1 --part-mib 1900 --paths \
  artifacts/bc_v1 artifacts/bc_v1_1 artifacts/bc_v1_2 reports/bc_v1_2 \
  reports/bc_interface_audit.json reports/bc_source_audit.json reports/bc_resources.json reports/bc_spawn_inventory.json reports/bc_semantics_audit.json \
  行为克隆实验汇总.html \
  logs/bc_v1_witness.log logs/bc_v1_witness_resources.jsonl logs/bc_state_diagnosis.log \
  logs/bc_v1_1_witness.log logs/bc_v1_1_witness_resources.jsonl \
  logs/bc_v1_1_interface.log logs/bc_v1_1_interface_resources.jsonl logs/bc_v1_1_existing_tests.log \
  logs/bc_v1_2_witness.log logs/bc_v1_2_witness_resources.jsonl logs/bc_v1_2_existing_tests.log \
  logs/bc_v1_2_collect.log logs/bc_v1_2_collect_resources.jsonl \
  logs/bc_v1_2_train.log logs/bc_v1_2_train_resources.jsonl \
  logs/bc_v1_2_report.log logs/bc_v1_2_sequence.log logs/bc_v1_2_sequence_exit.json
