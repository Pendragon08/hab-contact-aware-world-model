#!/usr/bin/env bash
# Finite R002 -> R003 continuation after verified R001 and fixed-spawn witness.
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
export HAB_BC_CONFIG=configs/bc_v1_2.json
python - <<'PY'
import json, hashlib, pathlib, shutil, os
root=pathlib.Path.cwd(); config=root/os.environ['HAB_BC_CONFIG']; plan=json.loads(config.read_text())
base=root/'artifacts'/plan['version']
witness=json.loads((base/'witness/complete.json').read_text()); assert witness['passed']
assert witness['plan_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
assert json.loads((root/plan['interface_source']/'complete.json').read_text())['passed']
snap=base/'source_snapshot'; snap.mkdir(exist_ok=False)
manifest=[]
for folder in ['scripts','configs','docs']:
    for path in sorted((root/folder).glob('*')):
        if not path.is_file() or path.suffix not in ('.py','.sh','.json','.md'):continue
        target=snap/folder/path.name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        manifest.append({'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(snap/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('BC_SOURCE_SNAPSHOT',len(manifest),flush=True)
PY
test ! -f logs/bc_v1_2_existing_tests.log
python -m unittest discover -s tests -v > logs/bc_v1_2_existing_tests.log 2>&1
for stage in collect train; do
  python - <<'PY'
import subprocess, shutil
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
cards={int(x[0]):[int(v.strip()) for v in x[1:]] for line in rows.splitlines() if (x:=line.split(','))}
assert cards[0][1]>=12288, 'GPU0 free reserve below user threshold; do not modify other processes'
assert cards[1][0]<500, 'GPU1 occupied by another process; stop, do not evict'
assert shutil.disk_usage('.').free>=15*2**30
print('STAGE_RESOURCE_CHECK',rows,flush=True)
PY
  bash scripts/run_bc.sh "$stage"
done
python scripts/report_bc.py > logs/bc_v1_2_report.log 2>&1
printf '%s\n' BC_SEQUENCE_FINISHED
