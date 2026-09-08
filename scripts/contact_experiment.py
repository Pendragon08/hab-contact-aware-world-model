"""Paired, seeded contact-control experiment with immutable raw observations."""
import argparse, hashlib, json, os, random, shutil, subprocess, time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from demo import create_env, load_policy, frame
from contact_control import ContactControl
from legacy_evaluate_layouts import CHECKERS, FLOATS, classify, cpu_tree, save_json, sha
from mshab.utils.array import to_tensor

ROOT=Path(os.environ['HAB_ROOT'])

def digest_tree(x):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor): v=v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            assert v.dtype != object
            h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v): h.update(str(k).encode()); visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v: visit(item)
        else: h.update(repr(v).encode())
    visit(x)
    return h.hexdigest()

def scalar(info,key,default=0.):
    return float(info[key][0].item()) if key in info else float(default)

def observe(obs, states, pixels):
    states.append(obs['state'][0].detach().cpu().numpy().copy())
    for key,value in obs['pixels'].items():
        # Save the newest of the 3 frames; initial stack is separately retained.
        pixels.setdefault(key,[]).append(value[0,-1].detach().cpu().numpy().copy())

def rollout(env, policy, plan, folder, layout, seed, pair_id, method, label=None, steps=200):
    if shutil.disk_usage(ROOT).free < 15*2**30: raise RuntimeError('Disk reserve below 15 GiB')
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    t_reset=time.perf_counter()
    obs,info=env.reset(seed=seed,options={'reconfigure':True,'build_config_idxs':[layout]})
    reset_seconds=time.perf_counter()-t_reset
    u=env.unwrapped
    assert list(u.build_config_idxs)==[layout]
    mapping={k:list(v) for k,v in u.agent.controller.action_mapping.items()}
    ctrl=ContactControl(method,mapping,plan['action'],plan['max_steps'])
    target=folder/(label or method)/f'episode_{pair_id:03d}'
    target.mkdir(parents=True,exist_ok=False)
    names={v:k for k,v in u.scene_builder.build_config_names_to_idxs.items()}
    scene={'build_config_idx':layout,'build_config_name':names[layout],
           'init_config_idxs':list(u.init_config_idxs),'task_plan_idxs':u.task_plan_idxs.cpu().tolist(),
           'spawn_selection_idxs':list(u.spawn_selection_idxs),'target_object':u.subtask_objs[0]._objs[0].name}
    snapshot={'environment_state':cpu_tree(u.get_state_dict()), 'controller_state':cpu_tree(u.agent.controller.get_state()),
              'observation':cpu_tree(obs),'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all(),
              'numpy_rng':np.random.get_state(), 'python_rng':random.getstate()}
    initial={'scene':scene,'state_sha256':digest_tree(snapshot['environment_state']),
             'controller_sha256':digest_tree(snapshot['controller_state']),
             'observation_sha256':digest_tree(snapshot['observation']),
             'initial_force':scalar(info,'robot_force'), 'initial_cumulative':scalar(info,'robot_cumulative_force')}
    torch.save(snapshot,target/'initial_snapshot.pt'); save_json(target/'initial.json',initial)
    save_json(target/'action_mapping.json',mapping)
    baseline=folder/'sac'/f'episode_{pair_id:03d}'/'initial.json'
    if (label or method) != 'sac':
        expected=json.loads(baseline.read_text())
        assert initial==expected, {'pair':pair_id,'method':method,'observed':initial,'expected':expected}
    writer=imageio.get_writer(target/'rollout.mp4',fps=20,codec='libx264',quality=7)
    events=[]; states=[]; pixels={}; actions=[]; actor_actions=[]
    writer.append_data(frame(env)); observe(obs,states,pixels)
    previous_force=scalar(info,'robot_force'); previous_cumulative=scalar(info,'robot_cumulative_force')
    started=time.perf_counter(); ended=False
    try:
        with (target/'steps.jsonl').open('w') as stream:
            for step in range(1,steps+1):
                obs=to_tensor(obs,device='cuda',dtype='float')
                torch.cuda.synchronize(); t=time.perf_counter()
                with torch.no_grad():
                    actor=policy.actor(obs['pixels'],obs['state'],compute_pi=False,compute_log_pi=False)[0]
                torch.cuda.synchronize(); actor_ms=(time.perf_counter()-t)*1000
                raw=actor[0].detach().cpu().numpy().copy()
                t=time.perf_counter()
                modified,metadata=ctrl(raw,previous_force,previous_cumulative,step)
                modifier_ms=(time.perf_counter()-t)*1000
                actor_actions.append(raw); actions.append(modified.copy())
                obs,reward,term,trunc,info=env.step(torch.from_numpy(modified).to('cuda')[None,:])
                ended=bool((term|trunc).any().item()); current=info.get('final_info',info)
                success=bool(info['episode']['s_e'][0].item()) if ended else bool(current['success'][0].item())
                event={'step':step,'success':success,'reward':float(reward[0].item()),
                       'actor_ms':actor_ms,'modifier_ms':modifier_ms,**metadata}
                for key in CHECKERS: event[key]=bool(current[key][0].item())
                for key in FLOATS: event[key]=scalar(current,key)
                assert success==all(event[k] for k in CHECKERS),event
                assert event['input_force']==previous_force and event['input_cumulative_force']==previous_cumulative
                event['object_xyz']=u.subtask_objs[0].pose.p[0].cpu().tolist()
                event['tcp_xyz']=u.agent.tcp_pose.p[0].cpu().tolist()
                event['rest_xyz']=u.ee_rest_world_pose.p[0].cpu().tolist()
                previous_force=event['robot_force']; previous_cumulative=event['robot_cumulative_force']
                stream.write(json.dumps(event)+'\n'); events.append(event)
                observe(obs,states,pixels); writer.append_data(frame(env))
                if ended:
                    terminal={k:v[0].item() for k,v in info['episode'].items()}
                    save_json(target/'terminal.json',terminal)
                    assert step==steps==200 and terminal['l']==200
                    assert bool(terminal['s_o'])==any(e['success'] for e in events)
                    assert bool(terminal['s_e'])==success
                    break
        assert len(events)==steps and (ended if steps==200 else not ended)
    finally: writer.close()
    data={'observed_states':np.stack(states),'actor_actions':np.stack(actor_actions),'executed_actions':np.stack(actions)}
    for k,v in pixels.items(): data[k]=np.stack(v)
    np.savez_compressed(target/'trajectory.npz',**data)
    first=next((e for e in events if e['success']),None)
    prefix=events[:first['step']] if first else events
    row={'pair_id':pair_id,'method':method,'seed':seed,'layout':layout,'steps':steps,
         'target_object':scene['target_object'],'success_once':first is not None,'success_at_end':events[-1]['success'],
         'force_exceeded_before_or_at_first_success':any(not e['cumulative_force_within_limit'] for e in prefix),
         'cumulative_force_at_first_success':first['robot_cumulative_force'] if first else None,
         'final_cumulative_force':events[-1]['robot_cumulative_force'],
         'return':sum(e['reward'] for e in events),'reset_seconds':reset_seconds,
         'rollout_seconds':time.perf_counter()-started,'actor_ms_mean':float(np.mean([e['actor_ms'] for e in events])),
         'modifier_ms_mean':float(np.mean([e['modifier_ms'] for e in events])),
         'mean_action_scale':float(np.mean([e['scale'] for e in events])),**classify(events)}
    save_json(target/'episode.json',row)
    print('EPISODE_COMPLETE',json.dumps(row),flush=True)
    return target

def main(args):
    plan_path=ROOT/'configs/contact_v1.json'; plan=json.loads(plan_path.read_text())
    folder=ROOT/'artifacts'/('contact_witness_v1' if args.mode=='witness' else 'contact_v1')
    folder.mkdir(parents=True,exist_ok=True)
    env=create_env()
    try:
        policy,policy_sha=load_policy(env); assert policy_sha==plan['checkpoint_sha256']
        spec=json.loads((ROOT/'configs/env-spec.json').read_text())
        spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
        assert spec_hash=='d873b21f'
        provenance={'plan_sha256':sha(plan_path),'checkpoint_sha256':policy_sha,'env_spec_hash':spec_hash,
                    'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'], 'torch':torch.__version__,
                    'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')}}
        save_json(folder/f'provenance_{args.method}.json',provenance)
        if args.mode=='run':
            witness=json.loads((ROOT/'artifacts/contact_witness_v1/complete.json').read_text())
            assert witness['passed'] and witness['plan_sha256']==sha(plan_path)
        rows=0
        for slot,layout in enumerate(plan['layouts']):
            for j in range(1 if args.mode=='witness' else plan['episodes_per_layout']):
                pair=slot*plan['episodes_per_layout']+j
                seed=(plan['witness_seed_base'] if args.mode=='witness' else plan['seed_base'])+slot*plan['seed_stride']+j
                methods=plan['methods'] if args.method=='all' else [args.method]
                for method in methods:
                    result=rollout(env,policy,plan,folder,layout,seed,pair,method,steps=30 if args.mode=='witness' else 200)
                    rows+=1
                    if args.mode=='witness' and method=='sac':
                        duplicate=rollout(env,policy,plan,folder,layout,seed,pair,'sac',label='sac_repeat',steps=30)
                        a=np.load(result/'trajectory.npz'); b=np.load(duplicate/'trajectory.npz')
                        diffs={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a.files}
                        save_json(duplicate/'repeatability.json',{'max_abs_differences':diffs,'tolerance':1e-5})
                        assert all(v<=1e-5 for v in diffs.values()),diffs
        marker={'passed':True,'mode':args.mode,'episodes':rows+(5 if args.mode=='witness' else 0),
                'plan_sha256':sha(plan_path),'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        save_json(folder/('complete.json' if args.method=='all' else f'complete_{args.method}.json'),marker)
        print('CONTACT_WITNESS_PASS' if args.mode=='witness' else 'CONTACT_RUN_COMPLETE',json.dumps(marker),flush=True)
    finally: env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['witness','run'],required=True)
    p.add_argument('--method',choices=['all','sac','conservative','force_rule'],default='all')
    main(p.parse_args())
