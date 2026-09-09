#!/usr/bin/env bash
set -euo pipefail
MW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAB_ROOT="$(dirname "$MW_ROOT")"
source "$HAB_ROOT/local-settings.sh"
export MW_ROOT HAB_ROOT HAB_PYTHON
export PYTHONPATH="$MW_ROOT/vendor_deps/v1:$MW_ROOT:$HAB_ROOT/vendor_tdmpc2"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export XDG_CACHE_HOME="$MW_ROOT/cache/xdg" TORCH_HOME="$MW_ROOT/cache/torch" TMPDIR="$MW_ROOT/tmp"
export MPLCONFIGDIR="$MW_ROOT/cache/matplotlib" CUDA_CACHE_PATH="$MW_ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$MW_ROOT/logs" "$MW_ROOT/artifacts"
cd "$HAB_ROOT"
