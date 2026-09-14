"""Post-run optimizer, initial-state isolation and paired outcome checks on CPU."""
import json
from pathlib import Path
import numpy as np
import torch
from bc_pipeline import ROOT, BASE, PLAN, BC

out=ROOT/'reports'/PLAN['version']
load=lambda p:json.loads(p.read_text())
ckpt=torch.load(BASE/'student/checkpoint_05000.pt',map_location='cpu',weights_only=False)
assert ckpt['update']==5000
steps=[int(s['step'].item()) for s in ckpt['optimizer']['state'].values()]
assert len(steps)==6 and all(s==5000 for s in steps)
assert all(torch.isfinite(v).all() for v in ckpt['model'].values())
torch.manual_seed(PLAN['training_seed'])
initial=BC(ckpt['model']['mean'],ckpt['model']['std'])
changes={k:float((v-ckpt['model'][k]).abs().max()) for k,v in initial.state_dict().items() if k.startswith('net.')}
assert all(v>0 for v in changes.values())
rows=load(BASE/'collection/episodes.json'); student=load(BASE/'student/episodes.json'); fit=rows[:80];dev=rows[80:]
assert {r['spawn_idx'] for r in fit}.isdisjoint({r['spawn_idx'] for r in dev})
assert len({r['spawn_idx'] for r in rows})==100
starts=[np.load(ROOT/r['folder']/'trajectory.npz')['states45'][0,:42] for r in rows]
distance=np.abs(np.asarray(starts[:80])[:,None]-np.asarray(starts[80:])[None]).max(-1)
assert (distance>1e-5).all()
tables={}
for key in ['ever_grasp','success_once','success_at_end']:
    tables[key]={f'teacher_{a}_student_{b}':sum(bool(t[key])==bool(a) and bool(s[key])==bool(b) for t,s in zip(dev,student))
                 for a in [0,1] for b in [0,1]}
    assert sum(tables[key].values())==20
result={'passed':True,'optimizer_parameter_steps':steps,'max_parameter_changes_from_seeded_initialization':changes,
        'unique_public_spawn_indices':100,'fit_dev_initial_state_min_linf_distance':float(distance.min()),
        'paired_outcomes':tables,'note':'One trained policy, 20 fixed development starts; tables do not estimate variation across training seeds or houses.'}
(out/'training_audit.json').write_text(json.dumps(result,indent=2))
print('BC_TRAINING_AUDIT_PASS',json.dumps(result))
