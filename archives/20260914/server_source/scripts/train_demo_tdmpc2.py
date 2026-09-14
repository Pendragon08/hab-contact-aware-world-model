"""Bounded demonstrated TD-MPC2 training with isolated, paired development tests."""
import argparse
import dataclasses
import json
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from bc_pipeline import create_train_env, native_state, seed_all
from bc_eval_v2 import canonical_pair
from contact_experiment import digest_tree
from demo import frame
from hab_state import ACTIVE
from legacy_evaluate_layouts import cpu_tree, CHECKERS, FLOATS
from train_state_pilot import td_config
from prepare_demo_tdmpc2 import sha, load, write
from demo_tdmpc2_agent import DemoReplay, DemoTDMPC2

ROOT = Path(os.environ['HAB_ROOT'])
CONFIG = ROOT/'configs/demo_tdmpc2_v1.json'
PLAN = load(CONFIG)
BASE = ROOT/'artifacts/demo_tdmpc2_v1'


def resource_guard(require_idle=False):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'],text=True)
    g = {int(a):[int(b),int(c)] for a,b,c in (line.split(',') for line in rows.splitlines())}
    assert g[0][1] >= 12288, g
    if require_idle:
        assert g[1][0] < 500,g
    assert shutil.disk_usage(ROOT).free >= 15*2**30, 'Disk reserve'
    return g


def rng_state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])


def reset(env, folder, index, save_snapshot):
    old = load(ROOT/'configs/bc_v1_2.json')
    seed,spawn = 1210000+index,old['spawn_index_order'][index]
    seed_all(seed)
    obs,info = env.reset(seed=seed,options={'reconfigure':True,'build_config_idxs':[6],
                         'task_plan_idxs':torch.tensor([456],dtype=torch.int),'spawn_selection_idxs':[spawn]})
    u = env.unwrapped
    assert list(u.build_config_idxs)==[6] and int(u.task_plan_idxs[0])==456 and list(u.spawn_selection_idxs)==[spawn]
    task = u.build_config_idx_to_task_plans[6][456]
    scene = {'layout':6,'task_idx':456,'init_config_idxs':list(u.init_config_idxs),
             'spawn_selection_idxs':list(u.spawn_selection_idxs),'target_object':u.subtask_objs[0]._objs[0].name,
             'task_plan':dataclasses.asdict(task)}
    assert task.subtasks[0].uid == PLAN['environment']['task_uid']
    snapshot = {'environment':cpu_tree(u.get_state_dict()),'controller':cpu_tree(u.agent.controller.get_state()),
                'observation':cpu_tree(obs)}
    initial = {'scene':scene,'environment_sha256':digest_tree(snapshot['environment']),
               'controller_sha256':digest_tree(snapshot['controller']),'observation_sha256':digest_tree(snapshot['observation'])}
    write(folder/'initial.json',initial)
    if save_snapshot:
        torch.save(snapshot,folder/'initial_snapshot.pt')
        write(folder/'initial_pair_audit.json',canonical_pair(ROOT/f'artifacts/bc_v1_2/collection/episode_{index:03d}',folder))
    x,_ = native_state(env,info,0)
    reference = np.load(ROOT/f'artifacts/bc_v1_2/collection/episode_{index:03d}/trajectory.npz')['states45'][0]
    np.testing.assert_allclose(x[0].cpu().numpy(),reference,atol=1e-5,rtol=0)
    return x,info


def physical(env):
    u = env.unwrapped
    return torch.cat([u.agent.robot.qpos,u.agent.robot.qvel,u.subtask_objs[0].pose.raw_pose,
                      u.agent.tcp_pose.raw_pose],1)[0].cpu().numpy().copy()


def episode(env, agent, folder, index, mpc, evaluation, update_callback=None, steps=200, replay_actions=None):
    resource_guard()
    folder.mkdir(parents=True,exist_ok=False)
    x,info = reset(env,folder,index,save_snapshot=evaluation)
    # Environment seeding must not reset the training/minibatch RNG every episode.
    # Caller restores its training RNG immediately after reset through this hook.
    if update_callback is not None:
        update_callback('after_reset',None)
    states=[x[0].cpu().numpy().copy()]; ps=[physical(env)]; actions=[]; rewards=[]; events=[]; latency=[]
    writer = imageio.get_writer(folder/'rollout.mp4',fps=20,codec='libx264',quality=6) if evaluation else None
    if writer:
        writer.append_data(frame(env))
    started=time.perf_counter()
    try:
        with (folder/'steps.jsonl').open('x',encoding='utf-8') as log:
            for t in range(steps):
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.no_grad():
                    action = agent.choose(x,t,mpc,evaluation) if replay_actions is None else torch.tensor(replay_actions[t:t+1])
                torch.cuda.synchronize();latency.append((time.perf_counter()-tick)*1000)
                action=action.to('cuda').reshape(1,11)
                assert torch.isfinite(action).all() and action.abs().max() <= 1.00001
                full=torch.zeros(1,13,device='cuda');full[:,ACTIVE]=action
                _,r,term,trunc,info=env.step(full)
                current=info.get('final_info',info)
                success=bool(info['episode']['s_e'][0]) if bool((term|trunc).any()) else bool(current['success'][0])
                e={'step':t+1,'reward':float(r[0]),'success':success}
                e.update({k:bool(current[k][0]) for k in CHECKERS})
                e.update({k:float(current[k][0]) for k in FLOATS})
                assert success == all(e[k] for k in CHECKERS)
                assert bool(trunc[0]) == (t==199)
                x,_=native_state(env,current,t+1)
                states.append(x[0].cpu().numpy().copy());ps.append(physical(env))
                actions.append(action[0].cpu().numpy().copy());rewards.append(float(r[0]));events.append(e)
                log.write(json.dumps(e)+'\n');log.flush()
                if writer:
                    writer.append_data(frame(env))
                if update_callback is not None:
                    update_callback('step',t)
    finally:
        if writer:
            writer.close()
        np.savez_compressed(folder/'trajectory.npz',states45=np.stack(states),physical_states=np.stack(ps),
                            effective_actions11=np.asarray(actions),rewards=np.asarray(rewards,dtype=np.float32),
                            action_latency_ms=np.asarray(latency))
    row={'collection_index':index,'seed':1210000+index,'spawn_idx':load(ROOT/'configs/bc_v1_2.json')['spawn_index_order'][index],
         'mode':'mpc' if mpc else 'direct_policy','steps':len(events),'ever_grasp':any(e['is_grasped'] for e in events),
         'success_once':any(e['success'] for e in events),'success_at_end':events[-1]['success'],
         'force_limit_exceeded_any':any(not e['cumulative_force_within_limit'] for e in events),
         'return':sum(rewards),'wall_seconds':time.perf_counter()-started,'latency_mean_ms':float(np.mean(latency)),
         'folder':folder.relative_to(ROOT).as_posix()}
    write(folder/'episode.json',row)
    if not evaluation and steps==200:
        agent.replay.add(np.stack(states),np.stack(actions),np.asarray(rewards,dtype=np.float32))
    print('EPISODE_COMPLETE',json.dumps(row),flush=True)
    return row


@torch.no_grad()
def prediction_audit(agent, folder):
    # Every model and persistence comparison uses the same windows (t >= 1).
    rows=[]
    for idx in range(80,100):
        src=ROOT/f'artifacts/bc_v1_2/collection/episode_{idx:03d}'
        with np.load(src/'trajectory.npz') as raw:
            obs=torch.tensor(raw['states45'],device='cuda')
            act=torch.tensor(raw['effective_actions11'],device='cuda')
        rewards=np.asarray([json.loads(s)['reward'] for s in (src/'steps.jsonl').read_text().splitlines()])
        starts=np.arange(1,198)
        z=agent.model.encode(agent.replay.normalize(obs[starts]),None)
        predictions=[]
        from common import math
        for h in range(3):
            a=act[starts+h]
            pr=math.two_hot_inv(agent.model.reward(z,a,None),agent.cfg).cpu().numpy().ravel()
            predictions.append(pr)
            target=rewards[starts+h];last=rewards[starts-1]
            rows.append({'collection_index':idx,'horizon':h+1,'windows':len(starts),
                         'model_mae':float(np.abs(pr-target).mean()),'persistence_mae':float(np.abs(last-target).mean()),
                         'model_rmse':float(np.sqrt(((pr-target)**2).mean())),
                         'persistence_rmse':float(np.sqrt(((last-target)**2).mean()))})
            z=agent.model.next(z,a,None)
        np.savez_compressed(folder/f'prediction_{idx:03d}.npz',start_indices=starts,predicted_rewards=np.stack(predictions),
                            actual_rewards=rewards)
    write(folder/'prediction_audit.json',{'rows':rows,'scope':'teacher development behavior; overlapping windows, not independent samples'})


def evaluate(env,agent,out,step,count=20):
    saved=rng_state();prior_mpc=agent.cfg.mpc
    prior={k:getattr(agent,k,None) for k in ['_prev_mean','_prev_mean_eval']}
    folder=out/f'evaluation_{step:05d}';folder.mkdir(exist_ok=False)
    rows=[]
    try:
        for index in range(80,80+count):
            for mpc in [False,True]:
                rows.append(episode(env,agent,folder/f"{'mpc' if mpc else 'direct_policy'}_{index:03d}",index,mpc,True))
                write(folder/'episodes.json',rows)
        prediction_audit(agent,folder)
        groups={}
        for mode in ['direct_policy','mpc']:
            rs=[r for r in rows if r['mode']==mode]
            groups[mode]={'episodes':len(rs),**{k:sum(r[k] for r in rs) for k in
                ['ever_grasp','success_once','success_at_end','force_limit_exceeded_any']}}
        write(folder/'complete.json',{'completed':True,'online_step':step,'counts':groups})
        print('PAIRED_EVALUATION_COMPLETE',step,json.dumps(groups),flush=True)
    finally:
        restore_rng(saved);agent.cfg.mpc=prior_mpc
        for key,value in prior.items():
            if value is None:
                if hasattr(agent,key):delattr(agent,key)
            else:setattr(agent,key,value)


def param_hashes(agent):
    import hashlib
    return {k:hashlib.sha256(torch.cat([p.detach().flatten() for p in module.parameters()]).cpu().numpy().tobytes()).hexdigest()
            for k,module in [('encoder',agent.model._encoder),('dynamics',agent.model._dynamics),
                             ('reward',agent.model._reward),('Q',agent.model._Qs),('policy',agent.model._pi)]}


def witness(env,agent,out):
    initial=param_hashes(agent)
    metrics=[agent.update_demo(True) for _ in range(10)]
    changed={k:initial[k]!=v for k,v in param_hashes(agent).items()}
    assert all(changed.values()),changed
    reference=ROOT/'artifacts/bc_v1_2/collection/episode_000'
    with np.load(reference/'trajectory.npz') as z:
        actions=z['effective_actions11'][:8]
        original=z['states45'][:9]
    row=episode(env,agent,out/'teacher_prefix',0,False,True,steps=8,replay_actions=actions)
    data=np.load(ROOT/row['folder']/'trajectory.npz')
    err=float(np.max(np.abs(data['states45']-original)))
    assert err<=1e-5,err
    expected=[json.loads(s)['reward'] for s in (reference/'steps.jsonl').read_text().splitlines()][:8]
    np.testing.assert_allclose(data['rewards'],expected,atol=1e-5,rtol=0)
    # A late-horizon planner must be invariant to actions beyond the last real step.
    z=agent.model.encode(agent.replay.normalize(torch.tensor(original[:1],device='cuda')),None).unsqueeze(1).repeat(1,4,1)
    a=torch.rand(1,3,4,11,device='cuda')*2-1
    agent.remaining=1
    v1=agent._estimate_value(z,a,None)
    a[:,1:]=-a[:,1:]
    v2=agent._estimate_value(z,a,None)
    torch.testing.assert_close(v1,v2,atol=0,rtol=0)
    terminal=agent.replay.base['terminal']
    assert terminal[:,-1].all() and not terminal[:,:-1].any()
    target=agent.finite_target(torch.ones(2,1,device='cuda'),torch.tensor([[True],[False]],device='cuda'),
                               torch.full((2,1),10.,device='cuda'))
    torch.testing.assert_close(target,torch.tensor([[1.],[10.75]],device='cuda'),atol=0,rtol=0)
    # Save/reload equivalence and both actual control paths.
    torch.save(agent.checkpoint(0,10),out/'checkpoint.pt')
    obs=torch.tensor(original[:1],device='cuda')
    saved=rng_state();a1=agent.choose(obs,0,False,True)
    ck=torch.load(out/'checkpoint.pt',weights_only=False)
    agent.model.load_state_dict(ck['model']);restore_rng(saved)
    torch.testing.assert_close(a1,agent.choose(obs,0,False,True),atol=0,rtol=0)
    for mpc in [False,True]:
        episode(env,agent,out/('mpc_prefix' if mpc else 'policy_prefix'),0,mpc,True,steps=8)
    # Synthetic online episode exercises mixed replay and bootstrap update code.
    p=agent.replay.base
    agent.replay.add(p['observations'][0].cpu().numpy(),p['actions'][0].cpu().numpy(),p['rewards'][0].cpu().numpy())
    metrics.append(agent.update_demo(False))
    write(out/'complete.json',{'passed':True,'changed_modules':changed,'teacher_prefix_state_error':err,
                              'metrics':metrics,'config_sha256':sha(CONFIG),'scope':'kernel/optimizer/interface witness; no performance claim'})
    print('DEMO_TDMPC2_WITNESS_PASS',changed,err,flush=True)


def run(env,agent,out):
    witness_meta=load(BASE/'witness/complete.json')
    assert witness_meta['passed'] and witness_meta['config_sha256']==sha(CONFIG)
    initial=param_hashes(agent)
    updates=0;start=time.perf_counter()
    with (out/'updates.jsonl').open('x',encoding='utf-8') as log:
        for i in range(PLAN['pretrain_updates']):
            m=agent.update_demo(True);updates+=1
            log.write(json.dumps({'stage':'pretrain','update':updates,'online_step':0,'seconds':time.perf_counter()-start,**m})+'\n')
            if updates%100==0:
                log.flush();resource_guard();print('PRETRAIN_UPDATE',updates,m,flush=True)
        agent.model._target_Qs.load_state_dict(agent.model._Qs.state_dict())
        assert all(initial[k]!=v for k,v in param_hashes(agent).items())
        torch.save(agent.checkpoint(0,updates),out/'checkpoint_00000.pt')
        evaluate(env,agent,out,0)
        # Episodes use only training spawn slots; round-robin covers all 80 before reuse.
        for ep in range(PLAN['online_steps']//200):
            saved=rng_state()
            def on_step(kind,t):
                nonlocal updates
                if kind=='after_reset':
                    restore_rng(saved)
                    return
                m=agent.update_demo(False);updates+=1
                step=ep*200+t+1
                log.write(json.dumps({'stage':'online','update':updates,'online_step':step,'seconds':time.perf_counter()-start,**m})+'\n')
                if step%100==0:
                    log.flush();print('ONLINE_UPDATE',step,m,flush=True)
            episode(env,agent,out/f'train_{ep:03d}',ep%80,True,False,update_callback=on_step)
            step=(ep+1)*200
            if step in PLAN['evaluation_online_steps']:
                torch.save(agent.checkpoint(step,updates),out/f'checkpoint_{step:05d}.pt')
                evaluate(env,agent,out,step)
    write(out/'complete.json',{'completed':True,'online_steps':PLAN['online_steps'],'pretrain_updates':PLAN['pretrain_updates'],
                              'updates':updates,'wall_seconds':time.perf_counter()-start,'final_parameter_hashes':param_hashes(agent),
                              'scope':'single training seed, one task, development evaluations only'})
    print('DEMO_TDMPC2_TRAINING_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['witness','run']);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    out=BASE/args.mode;out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();env=None
    def timeout(signum,frame):raise TimeoutError('Frozen stage time budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(PLAN['stage_seconds'][args.mode])
    try:
        g=resource_guard(require_idle=True)
        assert sha(ROOT/PLAN['data']['payload'])==PLAN['data']['sha256']
        paths=[CONFIG]+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'vendor_tdmpc2').rglob('*'))
        hashes={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}
        for rel in hashes:
            dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        write(out/'provenance.json',{'config_sha256':sha(CONFIG),'source_hashes':hashes,'environment_hash':'d873b21f',
                                     'resources_before':g,'training_seed':PLAN['training_seed']})
        seed_all(PLAN['training_seed'])
        with np.load(ROOT/PLAN['data']['payload']) as d:payload={k:d[k] for k in d.files}
        replay=DemoReplay(payload,PLAN['batch_size'],PLAN['horizon'])
        cfg=td_config(PLAN,45)
        cfg.env_id='PickSubtaskTrain-v0';cfg.steps=PLAN['online_steps'];cfg.seed=PLAN['training_seed']
        cfg.num_envs=1;cfg.num_eval_envs=1;cfg.exp_name=PLAN['version'];cfg.task_title=PLAN['environment']['task_uid']
        agent=DemoTDMPC2(cfg,replay,PLAN)
        OmegaConf.save(cfg,out/'resolved_config.yaml')
        env=create_train_env()
        if args.mode=='witness':witness(env,agent,out)
        else:run(env,agent,out)
        assert all(sha(ROOT/rel)==digest for rel,digest in hashes.items()),'Executing source changed'
        write(out/'process.json',{'completed':True,'seconds':time.perf_counter()-start,
                                 'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    except BaseException as exc:
        write(out/'process.json',{'completed':False,'seconds':time.perf_counter()-start,'error':repr(exc)})
        raise
    finally:
        signal.alarm(0)
        if env is not None:env.close()


if __name__=='__main__':main()
