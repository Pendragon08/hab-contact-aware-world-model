"""Summarize five-second whole-device samples; these are not continuous process peaks."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; records=[]; files=[]
for path in sorted((root/'logs').glob('bc_v*_resources.jsonl')):
    lines=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    files.append({'path':path.relative_to(root).as_posix(),'samples':len(lines)})
    for item in lines:
        if item['returncode'] != 0:continue
        for row in item['gpu_rows']:
            idx,used,free,util=[int(v.strip()) for v in row.split(',')]
            records.append({'gpu':idx,'used_mib':used,'free_mib':free,'util_percent':util,'unix':item['unix']})
summary={'sampling_interval_seconds':5,'scope':'whole GPUs, includes all processes; not a continuous allocation guarantee','files':files}
for idx in [0,1]:
    subset=[r for r in records if r['gpu']==idx]
    if subset:summary[f'gpu{idx}']={'samples':len(subset),'max_used_mib':max(r['used_mib'] for r in subset),
                                 'min_free_mib':min(r['free_mib'] for r in subset),'max_util_percent':max(r['util_percent'] for r in subset)}
summary['gpu0_sampled_reserve_passed']=summary['gpu0']['min_free_mib']>=12288
(root/'reports/bc_resources.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
