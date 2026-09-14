#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/dmc_tdmpc2_original/runtime.sh
export DMC_ROOT=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc
export XDG_CACHE_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/cache/xdg TORCH_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/cache/torch TMPDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/tmp
export MPLCONFIGDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/cache/matplotlib CUDA_CACHE_PATH=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/dmc
