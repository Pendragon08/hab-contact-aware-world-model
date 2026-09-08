#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
test -f artifacts/contact_v1/FINISHED
test -f artifacts/STATE_PILOTS_FINISHED
python -m unittest discover -s tests -v 2>&1 | tee logs/final_unit_tests.log
python scripts/verify_vendor.py
python scripts/verify_run_sources.py
python scripts/audit_observation_replay.py 2>&1 | tee logs/observation_replay_audit.log
python scripts/audit_parameter_counts.py
python scripts/audit_effective_config.py
python scripts/report_resources.py
date -u +%FT%TZ > artifacts/ALL_AUDITS_FINISHED
python scripts/archive_artifacts.py --name contact-v1 --paths artifacts/contact_v1
python scripts/archive_artifacts.py --name state-pilots-v1 --paths artifacts/state_tdmpc2_pilot_v1 artifacts/state_sac_pilot_v1 artifacts/STATE_PILOTS_FINISHED
python scripts/archive_artifacts.py --name witnesses-v1 --paths artifacts/contact_witness_v1 artifacts/state_tdmpc2_witness_v1 artifacts/state_sac_witness_v1
python scripts/archive_artifacts.py --name run-records-v1 --paths logs reports 接触控制对比汇总.html artifacts/ALL_AUDITS_FINISHED
date -u +%FT%TZ > release/READY
