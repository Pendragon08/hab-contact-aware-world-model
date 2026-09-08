#!/usr/bin/env bash
set -euo pipefail
source /data/code/ouyanghao/oyh/maniskill-hab-demo/scripts/runtime.sh
mkdir -p third_party/ManiSkill third_party/mshab inventory .aris/compute
tar -xf code_maniskill.tar -C third_party/ManiSkill
tar -xf code_mshab.tar -C third_party/mshab
cp -an /data/code/ouyanghao/oyh/world-model-risk-planning/cache/sapien/. cache/sapien/
"$HAB_PYTHON" -m pip install --target "$HAB_ROOT/vendor_deps" --no-deps shortuuid==1.0.13
"$HAB_PYTHON" -u scripts/download_assets.py
