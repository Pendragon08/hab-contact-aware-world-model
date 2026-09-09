"""Paper-era TD-MPC2, with local recording and an exact finite run budget.

Learning/planning/replay/environment wrappers are untouched upstream files.
The upstream train loop differs only in <= -> < to avoid an extra action.
"""
import os, sys, json, time, hashlib, inspect, textwrap, pathlib, argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from common import MODEL_SIZE
from common.seed import set_seed
from common.buffer import Buffer
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from envs.metaworld import make_env
from envs.wrappers.tensor import TensorWrapper

ROOT = pathlib.Path(os.environ['ORIG_ROOT'])

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')

def scalar(x):
    if torch.is_tensor(x): return float(x.detach().cpu())
    if isinstance(x, np.generic): return x.item()
    return x

def integrity():
    manifest=json.loads((ROOT/'upstream_manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'], row['path']
    return manifest['commit']

def config(steps=250000):
    c=OmegaConf.load(ROOT/'upstream/tdmpc2/config.yaml')
    del c['defaults']
    c.update({'task':'mw-pick-place','steps':steps,'seed':1,'model_size':5,
              'eval_freq':50000,'eval_episodes':10,'save_video':True,
              'multitask':False,'task_dim':0,'tasks':['mw-pick-place'],
              'bin_size':(c.vmax-c.vmin)/(c.num_bins-1)})
    for k,v in MODEL_SIZE[5].items(): c[k]=v
    return c

def environment(c):
    # The original V2 adapter executes each action twice and ends after 100 decisions.
    e=TensorWrapper(make_env(c))
    c.obs_shape={'state':e.observation_space.shape}
    c.action_dim=e.action_space.shape[0]
    c.episode_length=e.max_episode_steps
    c.seed_steps=max(1000,5*c.episode_length)
    assert tuple(c.obs_shape.state)==(39,) and c.action_dim==4 and c.episode_length==100
    return e

class RecordEnv:
    def __init__(self, env, out):
        self.env, self.out, self.phase=env,out,'train'
        self.counts={'train':0,'eval':0}; self.epcounts={'train':0,'eval':0}
        self.active=False
    def __getattr__(self,k): return getattr(self.env,k)
    def reset(self):
        assert not self.active, 'Unexpected reset of incomplete trajectory'
        obs=self.env.reset()
        self.rows={'obs':[obs.numpy().copy()],'action':[],'reward':[],'success':[]}
        self.active=True
        return obs
    def step(self,a):
        obs,r,d,info=self.env.step(a)
        self.counts[self.phase]+=1
        assert torch.isfinite(obs).all() and torch.isfinite(r), 'Nonfinite simulator output'
        assert not self.env.unwrapped._did_see_sim_exception, 'MuJoCo exception'
        self.rows['obs'].append(obs.numpy().copy()); self.rows['action'].append(a.numpy().copy())
        self.rows['reward'].append(float(r)); self.rows['success'].append(float(info['success']))
        if d:
            ix=self.epcounts[self.phase];self.epcounts[self.phase]+=1
            folder=self.out/'trajectories'/self.phase;folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'{ix:06d}.npz',**{k:np.array(v) for k,v in self.rows.items()})
            self.active=False
        return obs,r,d,info

class Video:
    def __init__(self,out): self.out=out;self.frames=[];self.enabled=False;self.context=None
    def init(self,env,enabled=True):
        self.enabled=enabled;self.frames=[]
        if enabled and self.context is None:
            import mujoco_py
            # Explicit physical GPU1 for EGL; CUDA_VISIBLE_DEVICES alone does not bind EGL.
            self.context=mujoco_py.MjRenderContextOffscreen(env.unwrapped.sim,device_id=1)
        self.record(env)
    def record(self,env):
        if self.enabled:self.frames.append(env.render())
    def save(self,step):
        if self.enabled:
            import imageio.v2 as imageio
            folder=self.out/'videos';folder.mkdir(exist_ok=True)
            imageio.mimsave(folder/f'eval_{step:06d}.mp4',self.frames,fps=15)

class Logger:
    def __init__(self,out,env,agent): self.out,self.env,self.agent=out,env,agent;self.video=Video(out)
    def log(self,d,category='train'):
        row={k:scalar(v) for k,v in d.items()};row['category']=category
        row['train_decisions']=self.env.counts['train'];row['train_native_steps']=2*self.env.counts['train']
        for k,v in row.items():
            if isinstance(v,float):assert np.isfinite(v),(k,v)
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if category=='eval':
            folder=self.out/'models';folder.mkdir(exist_ok=True)
            self.agent.save(folder/f"step_{int(d['step']):06d}.pt")
    def finish(self,agent):
        agent.save(self.out/'final.pt')

# Mechanical single-character bound correction; no rewrite of learning order.
train_source=textwrap.dedent(inspect.getsource(OnlineTrainer.train))
assert train_source.count('while self._step <= self.cfg.steps:')==1
train_source=train_source.replace('while self._step <= self.cfg.steps:','while self._step < self.cfg.steps:')
namespace={};exec(compile(train_source,'original_train_exact_budget','exec'),OnlineTrainer.train.__globals__,namespace)

class RecordedTrainer(OnlineTrainer):
    train=namespace['train']
    def eval(self):
        self.env.phase='eval'
        try:return super().eval()
        finally:self.env.phase='train'

def hashes(agent):
    modules={'encoder':agent.model._encoder,'dynamics':agent.model._dynamics,
             'reward':agent.model._reward,'Q':agent.model._Qs,'policy':agent.model._pi}
    return {k:hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in m.parameters())).hexdigest() for k,m in modules.items()}

def witness(out):
    c=config();set_seed(123);e=environment(c);a=TDMPC2(c);b=Buffer(c)
    # Use the real upstream transition packing, slice sampler and optimizer.
    t=OnlineTrainer(c,e,a,b,None);before=hashes(a);states=[]
    for ep in range(3):
        obs=e.reset();tds=[t.to_td(obs)];states.append(obs.numpy().tolist())
        for i in range(100):
            action=e.rand_act();obs,r,d,info=e.step(action);tds.append(t.to_td(obs,action,r))
            assert d==(i==99)
        b.add(torch.cat(tds))
    batch=list(b.sample())
    assert batch[0].shape==(4,256,39) and batch[1].shape==(3,256,4)
    assert all(torch.isfinite(x).all() for x in batch if x is not None)
    losses=[]
    for _ in range(5):
        loss={k:scalar(v) for k,v in a.update(b).items()}
        assert all(np.isfinite(x) for x in loss.values());losses.append(loss)
    after=hashes(a);assert all(before[k]!=after[k] for k in before)
    v=Video(out);obs=e.reset();v.init(e)
    for i in range(12):
        act=a.act(obs,t0=i==0,eval_mode=True);assert torch.isfinite(act).all()
        obs,r,d,info=e.step(act);v.record(e)
    v.save(0)
    # Fresh identical seeded environments must repeat one original reset and 20 fixed controls.
    traces=[]
    for _ in range(2):
        set_seed(777);ee=environment(c);o=ee.reset();trace=[o.numpy()]
        for i in range(20):o,_,_,_=ee.step(torch.tensor([.1,-.1,.1,0.]));trace.append(o.numpy())
        traces.append(np.array(trace));ee.close()
    error=float(np.max(np.abs(traces[0]-traces[1])));assert error<=1e-6,error
    result={'sentinel':'ORIGINAL_WITNESS_PASS','module_changed':{k:before[k]!=after[k] for k in before},
            'updates':5,'replay_batch_shape':list(batch[0].shape),'repeat_error':error,'losses':losses,
            'native_steps':704,'discount':a.discount,'gpu':torch.cuda.get_device_name(),
            'torch_peak_MiB':torch.cuda.max_memory_allocated()/2**20,'upstream_commit':integrity()}
    write_json(out/'witness.json',result);print('ORIGINAL_WITNESS_PASS',json.dumps(result),flush=True)

def run(out):
    c=config();set_seed(c.seed);env=RecordEnv(environment(c),out);agent=TDMPC2(c);buffer=Buffer(c)
    logger=Logger(out,env,agent);trainer=RecordedTrainer(c,env,agent,buffer,logger)
    OmegaConf.save(c,out/'resolved-config.yaml')
    started=time.time();trainer.train()
    assert env.counts['train']==250000 and not env.active
    # The original inclusive loop evaluated here, before its one extra action.
    result=trainer.eval();result.update(trainer.common_metrics());logger.log(result,'eval')
    write_json(out/'complete.json',{'train_decisions':env.counts['train'],
       'train_native_steps':2*env.counts['train'],'eval_decisions':env.counts['eval'],
       'reset_extra_native_steps':sum(env.epcounts.values()),'training_seed':c.seed,
       'final_eval':result,'seconds':time.time()-started,'upstream_commit':integrity(),
       'claim':'reduced-budget original-method reproduction; not full paper result reproduction'})
    print('ORIGINAL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['witness','train']);args=ap.parse_args()
    integrity();torch.backends.cudnn.benchmark=True
    out=ROOT/'artifacts'/('seed1_500k' if args.mode=='train' else 'witness_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    tracked=[ROOT/'runner.py',ROOT/'run.sh',ROOT/'runtime.sh',ROOT/'configs/env-spec.json']
    frozen={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in tracked}
    write_json(out/'executed-source.json',frozen)
    (witness if args.mode=='witness' else run)(out)
    assert frozen=={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in tracked}, 'Executing source changed'
