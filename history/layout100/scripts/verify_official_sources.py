"""Run locally: preserve upstream HF metadata separately from mirror downloads."""
import json, urllib.request
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name, repo, kind in [('pick_policy','arth-shukla/mshab_checkpoints','models'),
                         ('replicacad','haosulab/ReplicaCAD','datasets'),
                         ('rearrange','haosulab/ReplicaCADRearrange','datasets'),
                         ('ycb','haosulab/ManiSkill2','datasets')]:
    url=f'https://huggingface.co/api/{kind}/{repo}?blobs=true'
    with urllib.request.urlopen(url,timeout=30) as r: data=json.load(r)
    (root/'inventory'/f'{name}_official_metadata.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(name,data['sha'],flush=True)
