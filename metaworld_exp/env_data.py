"""Meta-World V3 native-reward trajectories with disjoint fixed reset seeds."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies import SawyerPickPlaceV3Policy

ROOT=Path(os.environ['MW_ROOT']);PLAN=json.loads((ROOT/'configs/pick_place_v1.json').read_text())
BASE=ROOT/'artifacts'/PLAN['version']


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def make_env(index,render=False):
    seed=PLAN['environment_seed_base']+index
    env=ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE['pick-place-v3-goal-observable'](seed=seed,render_mode='rgb_array' if render else None)
    env._freeze_rand_vec=False
    # V3 defaults to global NumPy RNG unless this documented environment switch is enabled.
    env.seeded_rand_vec=True
    obs,info=env.reset(seed=seed)
    assert obs.shape==(39,) and env.action_space.shape==(4,)
    return env,obs,info


def augment(obs,t):return np.r_[obs,t/500.].astype(np.float32)


def rollout(folder,index,actor=None,video=False,update=None,steps=500,restore_rng=None):
    folder.mkdir(parents=True,exist_ok=False)
    env,obs,info=make_env(index,video)
    expert=SawyerPickPlaceV3Policy()
    initial={'index':index,'seed':PLAN['environment_seed_base']+index,'observation':obs.tolist(),
             'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'randomization':env._last_rand_vec.tolist()}
    initial['randomization_sha256']=hashlib.sha256(env._last_rand_vec.tobytes()).hexdigest()
    write(folder/'initial.json',initial)
    teacher=BASE/f'collection/episode_{index:03d}/initial.json'
    if actor is not None and teacher.exists():
        ref=load(teacher)
        assert initial==ref,'Initial state differs from saved teacher'
    if restore_rng is not None:restore_rng()
    states=[augment(obs,0)];actions=[];labels=[];rewards=[];events=[];latencies=[]
    ps=[np.r_[env.data.qpos,env.data.qvel].copy()]
    writer=None;start=time.perf_counter()
    try:
        if video:
            import imageio.v2 as imageio
            writer=imageio.get_writer(folder/'rollout.mp4',fps=20,codec='libx264',quality=6)
            writer.append_data(env.render())
        with (folder/'steps.jsonl').open('x') as stream:
            for t in range(steps):
                label=np.clip(expert.get_action(obs),-1,1).astype(np.float32)
                tick=time.perf_counter()
                a=label.copy() if actor is None else np.asarray(actor(augment(obs,t),t),dtype=np.float32).reshape(4)
                latencies.append(1000*(time.perf_counter()-tick))
                assert np.isfinite(a).all() and np.abs(a).max()<=1.00001
                obs,r,term,trunc,info=env.step(a)
                assert not term and bool(trunc)==(t==499),(t,term,trunc)
                e={k:float(v) for k,v in info.items()};e.update(step=t+1,reward=float(r))
                assert e['success'] in (0.,1.) and np.isfinite(r)
                events.append(e);stream.write(json.dumps(e)+'\n')
                states.append(augment(obs,t+1));actions.append(a.copy());labels.append(label);rewards.append(r)
                ps.append(np.r_[env.data.qpos,env.data.qvel].copy())
                if writer:writer.append_data(env.render())
                if update is not None:update(t)
            stream.flush()
    finally:
        if writer:writer.close()
        env.close()
        np.savez_compressed(folder/'trajectory.npz',observations=np.asarray(states),actions=np.asarray(actions),
                            actor_labels=np.asarray(labels),rewards=np.asarray(rewards,dtype=np.float32),
                            physical_states=np.asarray(ps),latency_ms=np.asarray(latencies))
    row={'index':index,'seed':initial['seed'],'steps':len(events),'success_once':any(e['success'] for e in events),
         'success_at_end':bool(events[-1]['success']),'grasp_once':any(e.get('grasp_success',0) for e in events),
         'return':float(sum(rewards)),'latency_mean_ms':float(np.mean(latencies)),
         'folder':str(folder.relative_to(ROOT)),'trajectory_sha256':sha(folder/'trajectory.npz'),
         'randomization_sha256':initial['randomization_sha256'],'seconds':time.perf_counter()-start}
    write(folder/'episode.json',row)
    print('METAWORLD_EPISODE',json.dumps(row),flush=True)
    return row


def payload(rows,fixed_norm=None):
    values={k:[] for k in ['observations','actions','rewards','actor_labels']};success=[];correction=[]
    for row in rows:
        p=ROOT/row['folder']/'trajectory.npz';assert sha(p)==row['trajectory_sha256']
        assert 0<=row['index']<80 and row['steps']==500,'Training split violation'
        with np.load(p) as z:
            for k in values:values[k].append(z[k])
        corr='correction/' in row['folder']
        success.append(not corr and row['success_at_end']);correction.append(corr)
    d={k:np.stack(v) for k,v in values.items()}
    assert d['observations'].shape[1:]==(501,40) and d['actions'].shape[1:]==(500,4)
    d['terminal']=np.zeros_like(d['rewards'],dtype=bool);d['terminal'][:,-1]=True
    ret=np.zeros_like(d['rewards'],dtype=np.float64)
    for t in range(499,-1,-1):ret[:,t]=d['rewards'][:,t]+(PLAN['discount']*ret[:,t+1] if t<499 else 0)
    d['monte_carlo_returns']=ret.astype(np.float32)
    d['successful_teacher_episode']=np.array(success);d['correction_episode']=np.array(correction)
    assert sum(success)>=10,'Too few successful scripted teacher episodes'
    if fixed_norm is None:
        x=d['observations'][:,:-1].reshape(-1,40).astype(np.float64)
        d['mean']=x.mean(0).astype(np.float32);d['std']=np.maximum(x.std(0),.001).astype(np.float32)
        d['mean'][-1]=0;d['std'][-1]=1
    else:d['mean'],d['std']=fixed_norm
    return d


def collect():
    out=BASE/'collection';out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();rows=[]
    for index in range(80):
        assert time.perf_counter()-start<PLAN['stage_seconds']['collect']
        rows.append(rollout(out/f'episode_{index:03d}',index))
        write(out/'episodes.json',rows)
    assert len({r['randomization_sha256'] for r in rows})==80
    d=payload(rows);np.savez_compressed(out/'training.npz',**d)
    write(out/'complete.json',{'completed':True,'episodes':80,'control_steps':40000,
                              'success_once':sum(r['success_once'] for r in rows),'success_at_end':sum(r['success_at_end'] for r in rows),
                              'payload_sha256':sha(out/'training.npz'),'config_sha256':sha(ROOT/'configs/pick_place_v1.json')})
    print('METAWORLD_COLLECTION_COMPLETE',flush=True)


if __name__=='__main__':collect()
