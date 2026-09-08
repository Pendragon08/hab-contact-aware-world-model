"""Bounded online state-only training + separate validation, all transitions saved."""
import argparse,hashlib,json,os,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf
from hab_state import HABState
from episode_replay import EpisodeReplay
from state_sac import StateSAC

ROOT=Path(os.environ['HAB_ROOT']);sys.path.insert(0,str(ROOT/'vendor_tdmpc2'))
from common import MODEL_SIZE
from tdmpc2 import TDMPC2

CHECKERS=['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit']
def save(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def event(info,t,r):
    d={k:bool(info[k][0].item()) for k in CHECKERS};d.update(step=t,reward=float(r),success=bool(info['success'][0].item()))
    for k in ['robot_force','robot_cumulative_force','robot_target_pairwise_force']:d[k]=float(info[k][0].item())
    assert d['success']==all(d[k] for k in CHECKERS)
    return d
def rgb(env):
    x=env.env.unwrapped.render()
    if isinstance(x,torch.Tensor):x=x.cpu().numpy()
    return (x[0] if x.ndim==4 else x).astype(np.uint8)
def td_config(plan,obs_dim):
    cfg=OmegaConf.load(ROOT/'vendor_tdmpc2/config.yaml')
    for k,v in MODEL_SIZE[plan['model_size']].items():cfg[k]=v
    for k in ['model_size','batch_size','iterations','num_samples','num_elites','num_pi_trajs','horizon']:cfg[k]=plan[k]
    cfg.obs='state';cfg.obs_shape={'state':[obs_dim]};cfg.action_dim=11;cfg.multitask=False;cfg.task_dim=0
    cfg.episode_length=200;cfg.num_envs=1;cfg.num_eval_envs=1
    cfg.bin_size=(cfg.vmax-cfg.vmin)/(cfg.num_bins-1)
    return cfg

def main(args):
    plan=json.loads((ROOT/'configs/state_pilot_v1.json').read_text())
    torch.manual_seed(plan['seed']);np.random.seed(plan['seed']);random.seed(plan['seed'])
    out=ROOT/'artifacts'/f'state_{args.algo}_{args.mode}_v1';out.mkdir(parents=True,exist_ok=False)
    env=HABState('train');start=time.perf_counter();updates=0
    try:
        # All sampling uses actual train task plans, chosen without reading validation outcomes.
        obs,info=env.reset(plan['train_seed_base'])
        train_layouts=[env.layouts[i] for i in np.linspace(0,len(env.layouts)-1,5).round().astype(int)]
        dim=obs.shape[-1]
        agent=TDMPC2(td_config(plan,dim)) if args.algo=='tdmpc2' else StateSAC(dim,11)
        params=list(agent.model.parameters()) if args.algo=='tdmpc2' else list(agent.actor.parameters())
        initial_flat=torch.cat([p.detach().flatten() for p in params]).cpu().numpy()
        save(out/'provenance.json',{'plan_sha256':sha(ROOT/'configs/state_pilot_v1.json'),
             'source_hashes':{str(p.relative_to(ROOT)):sha(p) for folder in ['scripts','vendor_tdmpc2'] for p in (ROOT/folder).rglob('*.py')},
             'train_layouts':train_layouts,'training_split':'train','state_dim':dim,'active_action_dim':11,
             'trainable_model_parameters':sum(p.numel() for p in params if p.requires_grad),
             'gpu':torch.cuda.get_device_name(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
             'mode':args.mode,'initial_parameter_sha256':hashlib.sha256(initial_flat.tobytes()).hexdigest()})
        if args.algo=='tdmpc2':OmegaConf.save(agent.cfg,out/'resolved_config.yaml')
        replay=EpisodeReplay(plan['batch_size'],plan['horizon'])
        total=400 if args.mode=='witness' else plan['train_steps'];warmup=200 if args.mode=='witness' else plan['warmup_steps']
        assert total%200==0
        with (out/'updates.jsonl').open('w') as log:
            for ep in range(total//200):
                if shutil.disk_usage(ROOT).free<15*2**30:raise RuntimeError('Disk reserve')
                seed=plan['train_seed_base']+ep;layout=train_layouts[ep%len(train_layouts)]
                obs,info=env.reset(seed,layout)
                states=[obs[0].cpu().numpy().copy()];actions=[];rewards=[];events=[];latencies=[]
                initial_state=env.env.unwrapped.get_state_dict()
                torch.save({k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in initial_state.items()},out/f'train_{ep:03d}_initial.pt')
                for t in range(200):
                    g=ep*200+t
                    torch.cuda.synchronize();tick=time.perf_counter()
                    if g<warmup:a=torch.rand(1,11)*2-1
                    else:a=agent.act(obs,t0=t==0,eval_mode=False)
                    torch.cuda.synchronize();latencies.append(1000*(time.perf_counter()-tick))
                    nxt,r,done,info=env.step(a)
                    states.append(nxt[0].cpu().numpy().copy());actions.append(a[0].cpu().numpy().copy());rewards.append(float(r[0]))
                    events.append(event(info,t+1,r[0]));obs=nxt
                    if g>=warmup:
                        metrics=agent.update(replay);updates+=1
                        assert all(np.isfinite(float(v)) for v in metrics.values()),metrics
                        log.write(json.dumps({'step':g+1,'update':updates,'wall_seconds':time.perf_counter()-start,**metrics})+'\n')
                        if updates%100==0:log.flush();print('UPDATE_WITNESS',g+1,updates,metrics,flush=True)
                    assert done==(t==199)
                states=np.stack(states);actions=np.stack(actions);rewards=np.array(rewards,dtype=np.float32)
                replay.add(states,actions,rewards)
                np.savez_compressed(out/f'train_{ep:03d}.npz',observations=states,actions=actions,rewards=rewards,action_latency_ms=latencies)
                save(out/f'train_{ep:03d}.json',{'episode':ep,'seed':seed,'layout':layout,'steps':events,'split':'train'})
                print('TRAIN_EPISODE_COMPLETE',ep,ep*200+200,flush=True)
        final_flat=torch.cat([p.detach().flatten() for p in params]).cpu().numpy()
        assert np.isfinite(final_flat).all() and not np.array_equal(initial_flat,final_flat)
        train_seconds=time.perf_counter()-start
        if args.algo=='tdmpc2':
            agent.save(out/'policy.pt')
            torch.save({'model':agent.model.state_dict(),'optim':agent.optim.state_dict(),'pi_optim':agent.pi_optim.state_dict(),
                        'scale_state':agent.scale.state_dict() if hasattr(agent.scale,'state_dict') else vars(agent.scale)},out/'training_checkpoint.pt')
        else:torch.save(agent.checkpoint(),out/'policy.pt')
        env.close();env=None
        evaluation=HABState('val');eval_rows=[]
        try:
            for ep in range(2 if args.mode=='witness' else plan['eval_episodes']):
                layout=plan['eval_layouts'][ep%5];seed=plan['eval_seed_base']+ep
                obs,info=evaluation.reset(seed,layout);ev=[];states=[obs[0].cpu().numpy().copy()];acts=[]
                with imageio.get_writer(out/f'eval_{ep:03d}.mp4',fps=20,codec='libx264',quality=7) as writer:
                    writer.append_data(rgb(evaluation))
                    for t in range(200):
                        action=agent.act(obs,t0=t==0,eval_mode=True)
                        obs,r,done,info=evaluation.step(action)
                        ev.append(event(info,t+1,r[0]));states.append(obs[0].cpu().numpy().copy());acts.append(action[0].cpu().numpy().copy())
                        writer.append_data(rgb(evaluation))
                np.savez_compressed(out/f'eval_{ep:03d}.npz',observations=np.stack(states),actions=np.stack(acts))
                row={'episode':ep,'seed':seed,'layout':layout,'success_once':any(e['success'] for e in ev),'success_at_end':ev[-1]['success']}
                save(out/f'eval_{ep:03d}.json',{'summary':row,'steps':ev,'split':'val-development'});eval_rows.append(row)
        finally:evaluation.close()
        result={'completed':True,'algorithm':args.algo,'mode':args.mode,'train_steps':total,'updates':updates,
                'eval_steps':len(eval_rows)*200,'eval_episodes':len(eval_rows),'success_once_count':sum(r['success_once'] for r in eval_rows),
                'success_at_end_count':sum(r['success_at_end'] for r in eval_rows),'training_wall_seconds':train_seconds,
                'total_wall_seconds':time.perf_counter()-start,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'final_parameter_sha256':hashlib.sha256(final_flat.tobytes()).hexdigest(),
                'parameters_changed':True,'finite_updates':True,'scope':'engineering pilot, not convergence or final-test evidence'}
        save(out/'summary.json',result);print('STATE_PILOT_COMPLETE',json.dumps(result),flush=True)
    finally:
        if env is not None:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algo',choices=['tdmpc2','sac'],required=True);p.add_argument('--mode',choices=['witness','pilot'],required=True)
    main(p.parse_args())
