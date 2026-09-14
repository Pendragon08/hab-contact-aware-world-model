"""Summarize whole-card sampling, clearly separate it from PyTorch allocation."""
import json,os
from pathlib import Path
root=Path(os.environ['HAB_ROOT']);gpus={};sources={}
for path in sorted((root/'logs').glob('*resources.jsonl')):
    count=0
    for line in path.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row['returncode']!=0:continue
        for raw in row['gpu_rows']:
            index,used,free,util=[int(x.strip()) for x in raw.split(',')]
            d=gpus.setdefault(str(index),{'max_used_mib':0,'min_free_mib':10**9,'max_utilization_percent':0,'samples':0})
            d['max_used_mib']=max(d['max_used_mib'],used);d['min_free_mib']=min(d['min_free_mib'],free)
            d['max_utilization_percent']=max(d['max_utilization_percent'],util);d['samples']+=1
        count+=1
    sources[path.name]=count
result={'whole_card_gpu_samples':gpus,'files':sources,
        'gpu0_minimum_12_gib_reserve_observed':gpus['0']['min_free_mib']>=12*1024,
        'note':'Whole card, sampled every 5s, not per-process allocation nor a continuous peak guarantee. Contact and state pilots may overlap on GPU1. All experiment entrypoints set CUDA_VISIBLE_DEVICES=1.'}
out=root/'reports/resources.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
