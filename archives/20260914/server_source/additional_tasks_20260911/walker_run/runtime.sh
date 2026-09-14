#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/dmc_tdmpc2_original/runtime.sh
export DMC_ROOT=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run
export XDG_CACHE_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run/cache/xdg TORCH_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run/cache/torch TMPDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run/tmp
export MPLCONFIGDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run/cache/matplotlib CUDA_CACHE_PATH=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd /data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/walker_run
