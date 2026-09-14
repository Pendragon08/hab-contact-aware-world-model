#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/maniskill2_tdmpc2_original/runtime.sh
export MS_ROOT=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube
export XDG_CACHE_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube/cache/xdg TORCH_HOME=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube/cache/torch TMPDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube/tmp
export MPLCONFIGDIR=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube/cache/matplotlib CUDA_CACHE_PATH=/data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd /data/code/ouyanghao/oyh/hab-contact-aware-world-model/additional_tasks_20260911/lift_cube
