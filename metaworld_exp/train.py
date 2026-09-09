"""Isolated Meta-World pilot: demonstrations, correction, online MPC, paired tests."""
import argparse,hashlib,json,os,random,shutil,signal,subprocess,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from common import MODEL_SIZE
from agent import MetaWorldReplay,MetaWorldTDMPC2
from env_data import ROOT,BASE,PLAN,write,load,sha,rollout,payload


def guard(idle=False):
    s=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    g={int(a):(int(b),int(c)) for a,b,c in (x.split(',') for x in s.splitlines())}
    assert g[0][1]>=12288,g
    if idle:assert g[1][0]<500,g
    assert shutil.disk_usage(ROOT).free>=15*2**30,'Disk reserve'
    return g


def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def rng():return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()
def restore(x):random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state_all(x[3])


def config():
    cfg=OmegaConf.load(ROOT.parent/'vendor_tdmpc2/config.yaml')
    for k,v in MODEL_SIZE[PLAN['model_size']].items():cfg[k]=v
    for k in ['model_size','batch_size','horizon','iterations','num_samples','num_elites','num_pi_trajs']:cfg[k]=PLAN[k]
    cfg.obs='state';cfg.obs_shape={'state':[40]};cfg.action_dim=4;cfg.multitask=False;cfg.task_dim=0
    cfg.episode_length=500;cfg.num_envs=1;cfg.num_eval_envs=1
    cfg.bin_size=(cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    cfg.env_id=PLAN['task'];cfg.steps=PLAN['online_steps'];cfg.seed=PLAN['training_seed']
    cfg.exp_name=PLAN['version'];cfg.task_title=PLAN['task']
    return cfg


def hashes(agent):
    return {k:hashlib.sha256(torch.cat([p.detach().flatten() for p in m.parameters()]).cpu().numpy().tobytes()).hexdigest()
            for k,m in [('encoder',agent.model._encoder),('dynamics',agent.model._dynamics),('reward',agent.model._reward),('Q',agent.model._Qs),('policy',agent.model._pi)]}


def actor(agent,mpc,evaluation):
    def act(x,t):
        return agent.choose(torch.tensor(x[None],device='cuda'),t,mpc,evaluation).numpy()[0]
    return act


def evaluate(agent,out,step,final=False):
    saved=rng();prev={k:getattr(agent,k,None) for k in ['_prev_mean','_prev_mean_eval']};mode=agent.cfg.mpc
    folder=out/('final_test' if final else f'evaluation_{step:06d}');folder.mkdir()
    indices=range(*(PLAN['final_test_indices'] if final else PLAN['development_indices']))
    rows=[]
    train_hashes={r['randomization_sha256'] for r in load(BASE/'collection/episodes.json')}
    if final:
        train_hashes.update(r['randomization_sha256'] for r in load(out/'evaluation_000000/episodes.json'))
    try:
        for index in indices:
            guard()
            for mpc in [False,True]:
                name='mpc' if mpc else 'direct_policy'
                # Identical RNG seeds and saved physical initial states for paired starts.
                seed(1700000+index);state=rng()
                row=rollout(folder/f'{name}_{index:03d}',index,actor(agent,mpc,True),
                            video=index==indices.start,restore_rng=lambda:restore(state))
                row['mode']=name
                assert row['randomization_sha256'] not in train_hashes,'Split overlap'
                rows.append(row);write(folder/'episodes.json',rows)
            a=load(folder/f'direct_policy_{index:03d}/initial.json');b=load(folder/f'mpc_{index:03d}/initial.json')
            assert a==b,'Unpaired evaluation initial states'
        counts={}
        for name in ['direct_policy','mpc']:
            rs=[r for r in rows if r['mode']==name]
            counts[name]={'episodes':len(rs),**{k:sum(r[k] for r in rs) for k in ['grasp_once','success_once','success_at_end']},
                          'return_mean':float(np.mean([r['return'] for r in rs])),
                          'latency_mean_ms':float(np.mean([r['latency_mean_ms'] for r in rs]))}
        write(folder/'complete.json',{'completed':True,'online_step':step,'split':'final-test' if final else 'development','counts':counts})
        print('METAWORLD_PAIRED_EVALUATION',step,final,json.dumps(counts),flush=True)
    finally:
        restore(saved);agent.cfg.mpc=mode
        for k,v in prev.items():
            if v is None:
                if hasattr(agent,k):delattr(agent,k)
            else:setattr(agent,k,v)


def witness(agent,out):
    before=hashes(agent);metrics=[agent.update_demo(True) for _ in range(10)]
    changed={k:before[k]!=v for k,v in hashes(agent).items()};assert all(changed.values()),changed
    row=rollout(out/'teacher_prefix',0,video=True,steps=8)
    with np.load(ROOT/row['folder']/'trajectory.npz') as a,np.load(BASE/'collection/episode_000/trajectory.npz') as b:
        error=float(np.abs(a['observations']-b['observations'][:9]).max())
        np.testing.assert_allclose(a['rewards'],b['rewards'][:8],atol=1e-6,rtol=0)
        x=a['observations'][:1].copy()
    assert error<=1e-6,error
    with torch.no_grad():
        z=agent.model.encode(agent.replay.normalize(torch.tensor(x,device='cuda')),None).unsqueeze(1).repeat(1,4,1)
        a=torch.rand(1,3,4,4,device='cuda')*2-1;agent.remaining=1
        v1=agent._estimate_value(z,a,None);a[:,1:]=-a[:,1:];v2=agent._estimate_value(z,a,None)
        torch.testing.assert_close(v1,v2,atol=0,rtol=0)
    d=agent.replay.base['terminal'];assert d[:,-1].all() and not d[:,:-1].any()
    targets=agent.finite_target(torch.ones(2,1,device='cuda'),torch.tensor([[True],[False]],device='cuda'),torch.full((2,1),10.,device='cuda'))
    torch.testing.assert_close(targets,torch.tensor([[1.],[10.9]],device='cuda'))
    torch.save(agent.checkpoint(0,10),out/'checkpoint.pt')
    state=rng();a1=actor(agent,False,True)(x[0],0)
    ck=torch.load(out/'checkpoint.pt',weights_only=False);agent.model.load_state_dict(ck['model']);restore(state)
    np.testing.assert_array_equal(a1,actor(agent,False,True)(x[0],0))
    for mpc in [False,True]:rollout(out/('mpc_prefix' if mpc else 'policy_prefix'),0,actor(agent,mpc,True),video=True,steps=8)
    p=agent.replay.base
    agent.replay.add(p['observations'][0].cpu().numpy(),p['actions'][0].cpu().numpy(),p['rewards'][0].cpu().numpy())
    metrics.append(agent.update_demo(False))
    write(out/'complete.json',{'passed':True,'changed_modules':changed,'teacher_prefix_error':error,'metrics':metrics,
                              'config_sha256':sha(ROOT/'configs/pick_place_v1.json'),'controls':24,'optimizer_updates':11,
                              'scope':'kernel/render/optimizer/interface witness; not performance evidence'})
    print('METAWORLD_WITNESS_PASS',flush=True)


def run(agent,out):
    w=load(BASE/'witness/complete.json');assert w['passed'] and w['config_sha256']==sha(ROOT/'configs/pick_place_v1.json')
    assert load(ROOT/'reports/agent_follows_doc.json')['passed'],'Independent documented invocation required'
    updates=0;started=time.perf_counter();initial=hashes(agent)
    with (out/'updates.jsonl').open('x') as log:
        def update(pretrain,stage,step=0):
            nonlocal updates
            m=agent.update_demo(pretrain);updates+=1
            log.write(json.dumps({'stage':stage,'update':updates,'online_step':step,'seconds':time.perf_counter()-started,**m})+'\n')
            if updates%100==0:
                log.flush();print('METAWORLD_UPDATE',stage,updates,step,m,flush=True)
        for _ in range(PLAN['warmstart_updates']):update(True,'warmstart')
        torch.save(agent.checkpoint(0,updates),out/'warmstart.pt')
        teacher=load(BASE/'collection/episodes.json');corrections=[]
        for index in range(*PLAN['correction_indices']):
            guard();state=rng()
            corrections.append(rollout(out/f'correction/episode_{index:03d}',index,actor(agent,False,True),restore_rng=lambda:restore(state)))
            write(out/'correction/episodes.json',corrections)
        d=payload(teacher+corrections,(agent.replay.mean.cpu().numpy(),agent.replay.std.cpu().numpy()))
        np.savez_compressed(out/'training_with_corrections.npz',**d)
        agent.replay=MetaWorldReplay(d,PLAN['batch_size'],PLAN['horizon'])
        for _ in range(PLAN['joint_pretrain_updates']):update(True,'joint_pretrain')
        agent.model._target_Qs.load_state_dict(agent.model._Qs.state_dict())
        assert all(initial[k]!=v for k,v in hashes(agent).items())
        torch.save(agent.checkpoint(0,updates),out/'checkpoint_000000.pt');evaluate(agent,out,0)
        for ep in range(PLAN['online_steps']//500):
            guard();state=rng()
            row=rollout(out/f'online/episode_{ep:03d}',ep%80,actor(agent,True,False),
                        update=lambda t:update(False,'online',ep*500+t+1),restore_rng=lambda:restore(state))
            with np.load(ROOT/row['folder']/'trajectory.npz') as d:agent.replay.add(d['observations'],d['actions'],d['rewards'])
            step=(ep+1)*500
            if step in PLAN['evaluation_online_steps']:
                torch.save(agent.checkpoint(step,updates),out/f'checkpoint_{step:06d}.pt');evaluate(agent,out,step)
        evaluate(agent,out,PLAN['online_steps'],final=True)
    write(out/'complete.json',{'completed':True,'online_steps':PLAN['online_steps'],'updates':updates,
                              'wall_seconds':time.perf_counter()-started,'budgets':PLAN['budgets'],
                              'final_parameter_hashes':hashes(agent),'scope':'one training seed, single task V3 pilot; not MT1/MT50 or paper reproduction'})
    print('METAWORLD_TRAINING_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['witness','run']);args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    g=guard(idle=True)
    out=BASE/args.mode;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def timeout(*_):raise TimeoutError('Frozen stage time budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(PLAN['stage_seconds'][args.mode])
    try:
        source=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+list((ROOT/'configs').glob('*'))+list((ROOT.parent/'vendor_tdmpc2').rglob('*.py'))
        sources={str(p.relative_to(ROOT.parent)):sha(p) for p in source if p.is_file()}
        for rel in sources:
            dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT.parent/rel,dest)
        write(out/'provenance.json',{'source_hashes':sources,'resources_before':g,'training_seed':PLAN['training_seed'],'environment_hash':PLAN['environment_spec_hash']})
        c=load(BASE/'collection/complete.json');assert c['completed'] and c['payload_sha256']==sha(BASE/'collection/training.npz')
        assert c['config_sha256']==sha(ROOT/'configs/pick_place_v1.json')
        seed(PLAN['training_seed'])
        with np.load(BASE/'collection/training.npz') as d:data={k:d[k] for k in d.files}
        replay=MetaWorldReplay(data,PLAN['batch_size'],PLAN['horizon']);agent=MetaWorldTDMPC2(config(),replay,PLAN)
        OmegaConf.save(agent.cfg,out/'resolved_config.yaml')
        (witness if args.mode=='witness' else run)(agent,out)
        assert all(sha(ROOT.parent/rel)==v for rel,v in sources.items()),'Source modified during execution'
        write(out/'process.json',{'completed':True,'seconds':time.perf_counter()-started,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    except BaseException as exc:
        write(out/'process.json',{'completed':False,'seconds':time.perf_counter()-started,'error':repr(exc)});raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
