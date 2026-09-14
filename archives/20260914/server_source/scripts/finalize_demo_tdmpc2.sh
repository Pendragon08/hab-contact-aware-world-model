#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
test -f artifacts/demo_tdmpc2_v1/run/process.json
python scripts/report_demo_tdmpc2.py
python scripts/archive_artifacts.py --name demo-tdmpc2-v1 --paths artifacts/demo_tdmpc2_v1 reports/demo_tdmpc2_v1 示范辅助世界模型训练汇总.html logs/demo_tdmpc2_v1_witness.log logs/demo_tdmpc2_v1_witness_resources.jsonl logs/demo_tdmpc2_v1_run.log logs/demo_tdmpc2_v1_run_resources.jsonl
