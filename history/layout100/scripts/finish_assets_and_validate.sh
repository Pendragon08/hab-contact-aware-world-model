#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
tar -xf replicacad_remaining.tar -C assets/data/scene_datasets/replica_cad_dataset
"$HAB_PYTHON" scripts/finalize_staged_assets.py
"$HAB_PYTHON" scripts/verify_downloads.py
bash scripts/validate_env.sh
