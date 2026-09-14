#!/usr/bin/env bash
set -euo pipefail
MS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MS_ROOT/../local-settings.sh"
export MS_ROOT HAB_PYTHON
export PYTHONPATH="$MS_ROOT/vendor_deps:$MS_ROOT/../dmc_tdmpc2_original/vendor_deps:$MS_ROOT/../tdmpc2_original/vendor_deps/runtime:$MS_ROOT/../tdmpc2_original/upstream/tdmpc2"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1
export MS_RENDER_DEVICE=pci:65
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled LAZY_LEGACY_OP=0
export XDG_CACHE_HOME="$MS_ROOT/cache/xdg" TORCH_HOME="$MS_ROOT/cache/torch" TMPDIR="$MS_ROOT/tmp"
export MPLCONFIGDIR="$MS_ROOT/cache/matplotlib" CUDA_CACHE_PATH="$MS_ROOT/cache/cuda"
export MS2_ASSET_DIR="$MS_ROOT/assets"
mkdir -p "$TMPDIR" "$MS_ROOT/logs" "$MS_ROOT/artifacts" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$MS2_ASSET_DIR"
cd "$MS_ROOT"
