#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
mode="${1:?witness or run}"; method="${2:-all}"
case "$mode" in witness|run) ;; *) exit 2;; esac
python -m unittest discover -s tests -v
python scripts/contact_experiment.py --mode "$mode" --method "$method" 2>&1 | tee "logs/contact_${mode}_${method}.log"
