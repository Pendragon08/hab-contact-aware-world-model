#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/maniskill2_tdmpc2_original/runtime.sh
export MS_ROOT=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2
export XDG_CACHE_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2/cache/xdg TORCH_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2/cache/torch TMPDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2/tmp
export MPLCONFIGDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2/cache/matplotlib CUDA_CACHE_PATH=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd /data/code/ouyanghao/oyh/hab-contact-aware-world-model/throughput_trial/ms2
