#!/usr/bin/env bash
set -euo pipefail
DMC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DMC_ROOT/../local-settings.sh"
export DMC_ROOT HAB_PYTHON
export PYTHONPATH="$DMC_ROOT/vendor_deps:$DMC_ROOT/../tdmpc2_original/vendor_deps/runtime:$DMC_ROOT/../tdmpc2_original/upstream/tdmpc2"
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled LAZY_LEGACY_OP=0
export XDG_CACHE_HOME="$DMC_ROOT/cache/xdg" TORCH_HOME="$DMC_ROOT/cache/torch" TMPDIR="$DMC_ROOT/tmp"
export MPLCONFIGDIR="$DMC_ROOT/cache/matplotlib" CUDA_CACHE_PATH="$DMC_ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$DMC_ROOT/logs" "$DMC_ROOT/artifacts" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd "$DMC_ROOT"
