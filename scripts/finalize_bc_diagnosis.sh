#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source local-settings.sh
source scripts/runtime.sh
python scripts/report_bc_diagnosis.py
python - <<'PY'
import json,pathlib,shutil,subprocess
r=pathlib.Path.cwd();out=r/'reports/bc_diagnosis_v1'
records=[];files=[]
for p in sorted((r/'logs').glob('bc_diagnosis_v1_*_resources.jsonl')):
    data=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    files.append({'path':p.relative_to(r).as_posix(),'samples':len(data)})
    for s in data:
        if s['returncode']==0:
            for row in s['gpu_rows']:
                gpu,used,free,util=map(int,row.split(','));records.append((gpu,used,free,util))
summary={'files':files,'sampling_seconds':5,'scope':'Whole-card samples include all processes; transient unsampled peaks are not ruled out.'}
for gpu in [0,1]:
    data=[row for row in records if row[0]==gpu]
    summary[f'gpu{gpu}']={'max_used_mib':max(x[1] for x in data),'min_free_mib':min(x[2] for x in data)}
assert summary['gpu0']['min_free_mib']>=12288
(out/'resources.json').write_text(json.dumps(summary,indent=2))
(out/'gpu_after.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True))
dest=r/'artifacts/bc_diagnosis_v1/report_source';dest.mkdir(exist_ok=False)
for name in ['report_bc_diagnosis.py','finalize_bc_diagnosis.sh']:
    shutil.copyfile(r/'scripts'/name,dest/name)
print('BC_DIAGNOSIS_RESOURCES',json.dumps(summary),flush=True)
PY
python scripts/archive_artifacts.py --name bc-diagnosis-v1 --part-mib 1900 --paths \
  artifacts/bc_diagnosis_v1 reports/bc_diagnosis_v1 行为克隆失败诊断.html \
  docs/BC_DIAGNOSIS_EXECUTION.md configs/bc_diagnosis_v1.json logs/bc_diagnosis_v1_*
