"""CPU-only interface and scripted-policy probe; no rendering or CUDA kernels."""
import importlib.metadata as md
import json,os,time,hashlib
from pathlib import Path
import numpy as np
import metaworld,mujoco,gymnasium
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerPickPlaceV3Policy

root=Path(os.environ['MW_ROOT']);out=root/'artifacts/cpu_probe';out.mkdir(exist_ok=False)
rows=[];started=time.time()
for i in range(2):
    env=ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE['pick-place-v3-goal-observable'](seed=1500000)
    env._freeze_rand_vec=False
    obs,info=env.reset(seed=1500000)
    policy=SawyerPickPlaceV3Policy()
    states=[obs.copy()];actions=[];rewards=[];events=[]
    initial={'obs':obs.tolist(),'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'randomization':env._last_rand_vec.tolist()}
    for t in range(500):
        a=np.clip(policy.get_action(obs),-1,1).astype(np.float32)
        obs,r,term,trunc,info=env.step(a)
        assert not term and bool(trunc)==(t==499),(t,term,trunc)
        states.append(obs.copy());actions.append(a);rewards.append(r)
        events.append({k:float(v) for k,v in info.items()})
    assert np.shape(states)==(501,39) and np.shape(actions)==(500,4)
    arrays={'states':np.stack(states),'actions':np.stack(actions),'rewards':np.array(rewards)}
    np.savez_compressed(out/f'episode_{i}.npz',**arrays)
    (out/f'initial_{i}.json').write_text(json.dumps(initial))
    (out/f'events_{i}.json').write_text(json.dumps(events))
    rows.append({'success_once':any(e['success'] for e in events),'success_end':bool(events[-1]['success']),
                 'return':sum(rewards),'reward_min':min(rewards),'reward_max':max(rewards)})
    env.close()
a=np.load(out/'episode_0.npz');b=np.load(out/'episode_1.npz')
errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files};assert max(errors.values())<=1e-8,errors
versions={k:md.version(k) for k in ['metaworld','mujoco','gymnasium','numpy','torch','tensordict','etils','glfw','PyOpenGL']}
report={'passed':True,'seconds':time.time()-started,'versions':versions,'rows':rows,'duplicate_max_errors':errors,
        'observation_dim':39,'action_dim':4,'native_episode_steps':500,'gpu_kernel_and_render_tested':False}
(out/'complete.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
