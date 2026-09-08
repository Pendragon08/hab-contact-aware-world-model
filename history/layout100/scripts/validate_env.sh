#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
"$HAB_PYTHON" -u scripts/demo.py --validate --episodes 1 --output results/validation
