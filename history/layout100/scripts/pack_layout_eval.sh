#!/usr/bin/env bash
set -euo pipefail
cd /data/code/ouyanghao/oyh/maniskill-hab-demo
tar -czf staging/layout100_results_v1.tar.gz \
  results/pick_val_layout100_v1 \
  results/validation_layout100_v1 \
  results/validation_layout100_repair_v1 \
  configs/hab100_v1.json configs/hab100_v1_repair.json \
  logs/layout100_* \
  '跨布局评估汇总.html' layout100_manifest.json
sha256sum staging/layout100_results_v1.tar.gz
