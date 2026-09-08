"""Reconstruct stored depth stacks and verify frozen actor outputs off-simulator."""
import json,os
from pathlib import Path
import numpy as np
import torch
from demo import create_env,load_policy

root=Path(os.environ['HAB_ROOT']);env=create_env();rows=[]
try:
    actor,checkpoint=load_policy(env)
    # Fixed before looking at any outcome: first episode in each layout, four time points.
    for method in ['sac','conservative','force_rule']:
        for episode in [0,20,40,60,80]:
            path=root/f'artifacts/contact_v1/{method}/episode_{episode:03d}/trajectory.npz'
            with np.load(path) as data:
                for step in [0,29,99,199]:
                    pixels={k:torch.from_numpy(np.stack([data[k][max(0,step-j)] for j in [2,1,0]]))[None].float().cuda()
                            for k in ['fetch_head_depth','fetch_hand_depth']}
                    state=torch.from_numpy(data['observed_states'][step:step+1]).float().cuda()
                    with torch.no_grad():action=actor.actor(pixels,state,compute_pi=False,compute_log_pi=False)[0]
                    delta=float(np.abs(action[0].cpu().numpy()-data['actor_actions'][step]).max())
                    rows.append({'method':method,'episode':episode,'step_before_action':step,'max_action_error':delta})
                    assert delta<=1e-5,rows[-1]
finally:env.close()
out=root/'reports/contact_v1/observation_replay_audit.json'
out.write_text(json.dumps({'passed':True,'actor_calls':len(rows),'checkpoint_sha256':checkpoint,'max_action_error':max(r['max_action_error'] for r in rows),'checks':rows},indent=2))
print('OBSERVATION_REPLAY_PASS',len(rows),max(r['max_action_error'] for r in rows),flush=True)
