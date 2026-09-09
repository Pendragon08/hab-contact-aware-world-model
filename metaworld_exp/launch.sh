#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
test ! -e "$MW_ROOT/artifacts/pick_place_v1/run"
test -f "$MW_ROOT/reports/agent_follows_doc.json"
if screen -ls | grep -q '\.metaworld_pick_place_v1'; then
  echo 'Existing experiment session; refusing duplicate' >&2
  exit 1
fi
screen -dmS metaworld_pick_place_v1 bash -c 'bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/metaworld_exp/run.sh run && bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/metaworld_exp/finalize.sh > /data/code/ouyanghao/oyh/hab-contact-aware-world-model/metaworld_exp/logs/finalize.log 2>&1'
echo 'METAWORLD_SCREEN_DISPATCHED: inspect actual updates to verify launch'
