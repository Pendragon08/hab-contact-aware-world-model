"""Read-only status for the bounded BC diagnosis; does not load a GPU library."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'artifacts/bc_diagnosis_v1'
result={}
for stage in ['witness','fit','offline']:
    p=base/f'process_{stage}.json'
    if p.exists():result[stage]=json.loads(p.read_text())
p=base/'fit/episodes.json'
if p.exists():
    rows=json.loads(p.read_text())
    result['fit_episodes']={'completed':len(rows),**{k:sum(r[k] for r in rows) for k in ['ever_grasp','success_once','success_at_end']}}
p=base/'offline/episodes.json'
if p.exists():
    rows=json.loads(p.read_text());result['offline_progress']={'completed':len(rows),'last_group':rows[-1]['group'],'last_seed':rows[-1]['seed']}
print(json.dumps(result))
