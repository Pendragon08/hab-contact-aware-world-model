#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
test ! -e metaworld_exp/vendor_deps/v1
mkdir -p metaworld_exp/cache/wheels metaworld_exp/reports metaworld_exp/logs
export PIP_CACHE_DIR="$PWD/metaworld_exp/cache/pip" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
"$HAB_PYTHON" -m pip download --no-deps --dest metaworld_exp/cache/wheels -r metaworld_exp/configs/requirements-overlay.txt 2>&1 | tee metaworld_exp/logs/install_download.log
"$HAB_PYTHON" -m pip install --no-deps --no-index --find-links metaworld_exp/cache/wheels --target metaworld_exp/vendor_deps/v1 -r metaworld_exp/configs/requirements-overlay.txt 2>&1 | tee metaworld_exp/logs/install_overlay.log
"$HAB_PYTHON" - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('metaworld_exp')
spec=json.loads((p/'configs/env-spec.json').read_text())
digest=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
rows=[{'name':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in (p/'cache/wheels').glob('*.whl')]
(p/'reports/install.json').write_text(json.dumps({'spec_hash':digest,'wheels':rows,'ready':False,'reason':'kernel/physics/render/independent witness pending'},indent=2))
print('METAWORLD_OVERLAY_INSTALLED',digest,flush=True)
PY
