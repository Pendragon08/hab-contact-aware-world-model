"""Original TD-MPC2 learning core with the documented HAB task adapter."""
import os,json,time,pathlib,hashlib,inspect,textwrap,subprocess,shutil,argparse
import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf
from hab_state import HABState
from common import MODEL_SIZE
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

ROOT=pathlib.Path(os.environ['HAB_ROOT']); HERE=ROOT/'hab_tdmpc2_original'
PLAN=json.loads((HERE/'configs/plan.json').read_text())
CHECKERS=['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit']

def save(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n')
def scalar(x):
    if torch.is_tensor(x):return float(x.detach().cpu())
    if isinstance(x,np.generic):return x.item()
    return x
def guard(witness=False):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    g={int(x[0]):x[1:] for x in [line.split(',') for line in rows.splitlines()]}
    assert int(g[0][2])>=12288 and int(g[1][2])>=12288,g
    assert shutil.disk_usage(ROOT).free>=15*2**30,'Disk reserve'
    if witness:
        # A bounded witness can share GPU1 only with our known Meta-World run.
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        for line in apps.splitlines():
            pid,uuid=[x.strip() for x in line.split(',')]
            if uuid==g[1][0].strip():
                args=subprocess.check_output(['ps','-p',pid,'-o','args='],text=True)
                owner=subprocess.check_output(['ps','-p',pid,'-o','user='],text=True).strip()
                assert owner==os.environ['USER'] and str(ROOT/'tdmpc2_original/runner.py')+' train' in args,(pid,args)
    else:assert int(g[1][1])<500,g
    return rows

def verify_sources():
    p=ROOT/'tdmpc2_original'
    for row in json.loads((p/'upstream_manifest.json').read_text())['files']:
        assert hashlib.sha256((p/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']

class Env:
    def __init__(self,out):
        self.out=out;self.phase='train';self.actual=None;self.env=None;self.active=False
        self.counts={'train':0,'eval':0};self.episodes={'train':0,'eval':0};self.eval_index=0
        self.eval_stage=0;self.rows=[];self.elapsed=0;self.train_layouts=None
    def rand_act(self):return torch.rand(11)*2-1
    def reset(self):
        assert not self.active
        split='train' if self.phase=='train' else 'val'
        if split!=self.actual:
            if self.env is not None:self.env.close()
            self.env=HABState(split);self.actual=split
        if split=='train':
            if self.train_layouts is None:
                self.train_layouts=[self.env.layouts[i] for i in np.linspace(0,len(self.env.layouts)-1,PLAN['train_layout_count']).round().astype(int)]
                save(self.out/'train-layouts.json',self.train_layouts)
            i=self.episodes['train'];layout=self.train_layouts[i%len(self.train_layouts)];seed=PLAN['train_seed_base']+i
        else:
            i=self.eval_index;self.eval_index+=1;layout=PLAN['validation_layouts'][i%5];seed=PLAN['eval_seed_base']+i
        x,info=self.env.reset(seed,layout)
        self.active=True;self.elapsed=0;self.seed=seed;self.layout=layout
        self.obs=[x[0,:-1].cpu().numpy().copy()];self.actions=[];self.rewards=[];self.events=[]
        assert len(self.obs[0])==44
        return torch.from_numpy(self.obs[0])
    def step(self,a):
        x,r,done,info=self.env.step(a)
        self.elapsed+=1;self.counts[self.phase]+=1
        obs=x[0,:-1].cpu();reward=r[0].detach().cpu()
        assert torch.isfinite(obs).all() and torch.isfinite(reward)
        e={k:bool(info[k][0]) for k in CHECKERS};e['success']=bool(info['success'][0])
        assert e['success']==all(e[k] for k in CHECKERS)
        e.update({k:float(info[k][0]) for k in ['robot_force','robot_cumulative_force']})
        e.update(step=self.elapsed,reward=float(reward))
        self.obs.append(obs.numpy().copy());self.actions.append(a.numpy().copy());self.rewards.append(float(reward));self.events.append(e)
        if done:
            assert self.elapsed==200
            folder=self.out/'trajectories'/self.phase;folder.mkdir(parents=True,exist_ok=True)
            index=self.episodes[self.phase];self.episodes[self.phase]+=1
            stem=f'{index:06d}'
            np.savez_compressed(folder/(stem+'.npz'),obs=np.array(self.obs),action=np.array(self.actions),reward=np.array(self.rewards))
            row={'seed':self.seed,'layout':self.layout,'stage':self.eval_stage,'success_once':any(v['success'] for v in self.events),'success_at_end':e['success'],'events':self.events}
            save(folder/(stem+'.json'),row);self.active=False
            if self.phase=='eval':self.rows.append(row)
            if shutil.disk_usage(ROOT).free<15*2**30:raise RuntimeError('Disk reserve')
        return obs,reward,done,{'success':float(e['success'])}
    def close(self):
        if self.env is not None:self.env.close();self.env=None

class Video:
    def __init__(self,out):self.out=out;self.enabled=False
    def init(self,env,enabled=True):
        self.enabled=enabled;self.frames=[];self.record(env)
    def record(self,env):
        if self.enabled:
            f=env.env.env.unwrapped.render()
            if torch.is_tensor(f):f=f.cpu().numpy()
            self.frames.append((f[0] if f.ndim==4 else f).astype(np.uint8))
    def save(self,step):
        if self.enabled:
            p=self.out/'videos';p.mkdir(exist_ok=True)
            imageio.mimsave(p/f'eval_{step:06d}.mp4',self.frames,fps=20)

class Logger:
    def __init__(self,out,env,agent):self.out,self.env,self.agent=out,env,agent;self.video=Video(out)
    def log(self,d,category='train'):
        row={k:scalar(v) for k,v in d.items()};row['category']=category
        row.update(train_steps=self.env.counts['train'],eval_steps=self.env.counts['eval'])
        assert all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if category=='eval':
            p=self.out/'models';p.mkdir(exist_ok=True);self.agent.save(p/f"step_{d['step']:06d}.pt")
    def finish(self,agent):agent.save(self.out/'final.pt')

source=textwrap.dedent(inspect.getsource(OnlineTrainer.train))
assert source.count('while self._step <= self.cfg.steps:')==1
source=source.replace('while self._step <= self.cfg.steps:','while self._step < self.cfg.steps:')
ns={};exec(compile(source,'original_train_exact_budget','exec'),OnlineTrainer.train.__globals__,ns)
class Trainer(OnlineTrainer):
    train=ns['train']
    def eval(self):
        # Preserve training RNG across separate held-out-layout evaluation.
        rng=(np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
        self.env.phase='eval';self.env.eval_index=0;self.env.eval_stage=self._step;start=len(self.env.rows)
        try:
            d=super().eval();d['success_once']=np.mean([r['success_once'] for r in self.env.rows[start:]])
            return d
        finally:
            self.env.phase='train';np.random.set_state(rng[0]);torch.set_rng_state(rng[1]);torch.cuda.set_rng_state(rng[2])

def hashes(a):
    modules=[a.model._encoder,a.model._dynamics,a.model._reward,a.model._Qs,a.model._pi]
    return [hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.parameters())).hexdigest() for m in modules]

def main(mode):
    resources=guard(mode=='witness');verify_sources();set_seed(PLAN['seed'])
    out=HERE/'artifacts'/('seed1_500k' if mode=='train' else 'witness_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    frozen={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'runner.py',HERE/'configs/plan.json',HERE/'configs/env-spec.json',ROOT/'scripts/hab_state.py']}
    save(out/'sources.json',frozen)
    c=OmegaConf.load(ROOT/'tdmpc2_original/upstream/tdmpc2/config.yaml');del c['defaults']
    for k,v in MODEL_SIZE[5].items():c[k]=v
    c.update({'task':'hab-tidyhouse-pick','model_size':5,'obs':'state','obs_shape':{'state':[44]},'action_dim':11,'multitask':False,'task_dim':0,'tasks':['hab-tidyhouse-pick'],
      'episode_length':200,'seed_steps':1000,'steps':1200 if mode=='witness' else PLAN['steps'],
      'eval_freq':PLAN['eval_freq'],'eval_episodes':1 if mode=='witness' else PLAN['eval_episodes'],
      'batch_size':256,'bin_size':(c.vmax-c.vmin)/(c.num_bins-1),'seed':PLAN['seed'],'save_video':True})
    agent=TDMPC2(c);before=hashes(agent);env=Env(out);logger=Logger(out,env,agent);buffer=Buffer(c)
    t=Trainer(c,env,agent,buffer,logger);OmegaConf.save(c,out/'resolved-config.yaml')
    start=time.time()
    update_count=0
    original_update=agent.update
    def checked_update(replay):
        nonlocal update_count
        metrics={k:scalar(v) for k,v in original_update(replay).items()}
        assert all(np.isfinite(v) for v in metrics.values()),metrics
        update_count+=1
        if update_count==1 or update_count%100==0:
            row={'update':update_count,'train_steps':env.counts['train'],'seconds':time.time()-start,**metrics}
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('HAB_ORIGINAL_UPDATE',json.dumps(row),flush=True)
        return metrics
    agent.update=checked_update
    try:
        t.train();train_seconds=time.time()-start
        assert env.counts['train']==c.steps and not env.active
        after=hashes(agent);assert all(a!=b for a,b in zip(before,after))
        ev={k:scalar(v) for k,v in t.eval().items()};ev.update(t.common_metrics());logger.log(ev,'eval')
        verify_sources()
        for n,h in frozen.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
        result={'completed':True,'mode':mode,'train_steps':env.counts['train'],'optimizer_updates':update_count,'eval_steps':env.counts['eval'],'final_eval':ev,
           'all_five_modules_changed':True,'discount':agent.discount,'train_seconds_including_initial_eval':train_seconds,
           'total_seconds':time.time()-start,'torch_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'preflight_gpu':resources,
           'scope':'HAB transfer of original TD-MPC2, one seed; not original paper HAB result'}
        save(out/'complete.json',result)
        print('HAB_ORIGINAL_WITNESS_PASS' if mode=='witness' else 'HAB_ORIGINAL_TRAINING_COMPLETE',json.dumps(result),flush=True)
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['witness','train']);main(p.parse_args().mode)
