#!/usr/bin/env bash
set -euo pipefail
ORIG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ORIG_ROOT/../local-settings.sh"
export ORIG_ROOT HAB_PYTHON
export PYTHONPATH="$ORIG_ROOT/vendor_deps/runtime:$ORIG_ROOT/vendor_deps/Metaworld-04be337a12305e393c0caf0cbf5ec7755c7c8feb:$ORIG_ROOT/upstream/tdmpc2"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_PY_MUJOCO_PATH="$ORIG_ROOT/native/mujoco210"
export CPATH="$ORIG_ROOT/native/sysroot/usr/include:${CPATH:-}"
export MUJOCO_PY_NVIDIA_LIB_DIR=/usr/lib/x86_64-linux-gnu
export PATH="$ORIG_ROOT/vendor_deps/runtime/bin:$PATH"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export XDG_CACHE_HOME="$ORIG_ROOT/cache/xdg" TORCH_HOME="$ORIG_ROOT/cache/torch" TMPDIR="$ORIG_ROOT/tmp"
export MPLCONFIGDIR="$ORIG_ROOT/cache/matplotlib" CUDA_CACHE_PATH="$ORIG_ROOT/cache/cuda"
mkdir -p "$TMPDIR" "$ORIG_ROOT/logs" "$ORIG_ROOT/artifacts"
cd "$ORIG_ROOT/.."
