import torch,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'artifacts/bc_v1_2'
a=torch.load(root/'collection/episode_080/initial_snapshot.pt',map_location='cpu',weights_only=False)['environment']['actors']
b=torch.load(root/'student/evaluation/episode_080/initial_snapshot.pt',map_location='cpu',weights_only=False)['environment']['actors']
rows=[]
for k in a:
    if not torch.equal(a[k],b[k]):rows.append({'name':k,'teacher':a[k].tolist(),'student':b[k].tolist()})
(root/'parked_actor_diagnosis.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows[:4],indent=2));print('ACTORS',len(a),'DIFFERENT',len(rows))
