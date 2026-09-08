#!/usr/bin/env bash
set -euo pipefail
HAB_ROOT=/data/code/ouyanghao/oyh/maniskill-hab-demo
HAB_PYTHON=/data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$HAB_ROOT/vendor_deps:$HAB_ROOT/third_party/ManiSkill:$HAB_ROOT/third_party/mshab"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MS_ASSET_DIR="$HAB_ROOT/assets"
export HF_HOME="$HAB_ROOT/cache/huggingface" HF_HUB_DISABLE_XET=1
export WM_SAPIEN_CACHE="$HAB_ROOT/cache/sapien"
export XDG_CACHE_HOME="$HAB_ROOT/cache/xdg" TORCH_HOME="$HAB_ROOT/cache/torch"
export TMPDIR="$HAB_ROOT/tmp" MPLCONFIGDIR="$HAB_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$HAB_ROOT/cache/cuda"
export PIP_CACHE_DIR="$HAB_ROOT/cache/pip" TORCH_EXTENSIONS_DIR="$HAB_ROOT/cache/torch_extensions"
export PATH="$(dirname "$HAB_PYTHON"):$PATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$HAB_ROOT/logs" "$HAB_ROOT/results" "$HAB_ROOT/vendor_deps" "$WM_SAPIEN_CACHE"
cd "$HAB_ROOT"
export HAB_ROOT HAB_PYTHON
