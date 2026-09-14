#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/runtime.sh
exec "$HAB_PYTHON" /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/runner.py train --seed 2
