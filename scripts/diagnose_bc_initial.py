"""Compare the retained first teacher/student initial snapshots on CPU, without rerunning."""
import json
from pathlib import Path
import numpy as np
import torch
root=Path(__file__).resolve().parents[1]; base=root/'artifacts/bc_v1_2'
a=base/'collection/episode_080';b=base/'student/evaluation/episode_080'
load=lambda p:json.loads(p.read_text())
x=torch.load(a/'initial_snapshot.pt',map_location='cpu',weights_only=False)
y=torch.load(b/'initial_snapshot.pt',map_location='cpu',weights_only=False)
rows=[]
def compare(aa,bb,path):
    if isinstance(aa,dict):
        assert aa.keys()==bb.keys(),path
        for k in aa:compare(aa[k],bb[k],path+'.'+str(k))
    elif isinstance(aa,(list,tuple)):
        assert len(aa)==len(bb)
        for i,(av,bv) in enumerate(zip(aa,bb)):compare(av,bv,path+f'[{i}]')
    elif isinstance(aa,(torch.Tensor,np.ndarray)):
        av=aa.numpy() if isinstance(aa,torch.Tensor) else aa; bv=bb.numpy() if isinstance(bb,torch.Tensor) else bb
        assert av.shape==bv.shape,path
        diff=float(np.max(np.abs(av.astype(float)-bv.astype(float)))) if av.size else 0.
        rows.append({'path':path,'shape':list(av.shape),'max_abs_difference':diff,'dtype':str(av.dtype)})
    else:
        if aa!=bb:rows.append({'path':path,'a':aa,'b':bb,'unequal':True})
for k in ['environment','controller','observation']:compare(x[k],y[k],k)
result={'teacher_initial':load(a/'initial.json'),'student_initial':load(b/'initial.json'),
        'differences':[r for r in rows if r.get('max_abs_difference',0)>0 or r.get('unequal')],
        'all_checks':rows,'student_episode':load(b/'episode.json')}
(base/'initial_pair_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='all_checks'},indent=2))
