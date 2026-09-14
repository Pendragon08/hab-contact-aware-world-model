"""Three fixed-seed, inference-only demos; never change the original score."""
import json,hashlib,pathlib,time,subprocess,os
import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf
from common.seed import set_seed
from tdmpc2 import TDMPC2
from envs.metaworld import make_env
from envs.wrappers.tensor import TensorWrapper
import mujoco_py

root=pathlib.Path(os.environ['ORIG_ROOT'])
checkpoint=root/'artifacts/seed1_500k/final.pt'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=digest(checkpoint)
free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
assert free>12288
out=root/'artifacts/final_demo_seed101_103';out.mkdir(exist_ok=False)
cfg=OmegaConf.load(root/'artifacts/seed1_500k/resolved-config.yaml')
rows=[]
for seed in [101,102,103]:
    set_seed(seed);cfg.seed=seed
    env=TensorWrapper(make_env(cfg));agent=TDMPC2(cfg);agent.load(checkpoint)
    context=mujoco_py.MjRenderContextOffscreen(env.unwrapped.sim,device_id=1)
    obs=env.reset();frames=[env.render()];states=[obs.numpy().copy()];actions=[];rewards=[];success=[]
    start=time.time()
    for step in range(100):
        with torch.no_grad():action=agent.act(obs,t0=step==0,eval_mode=True)
        obs,reward,done,info=env.step(action)
        assert done==(step==99)
        frames.append(env.render());states.append(obs.numpy().copy());actions.append(action.numpy().copy())
        rewards.append(float(reward));success.append(float(info['success']))
    video=out/f'seed_{seed}.mp4';imageio.mimsave(video,frames,fps=15)
    np.savez_compressed(out/f'seed_{seed}.npz',obs=np.array(states),action=np.array(actions),reward=np.array(rewards),success=np.array(success))
    row={'seed':seed,'success_at_end':bool(success[-1]),'success_once':bool(max(success)),
         'reward':sum(rewards),'decisions':100,'native_steps':200,'seconds':time.time()-start,
         'video':video.name,'video_sha256':digest(video),'video_bytes':video.stat().st_size}
    rows.append(row);print(json.dumps(row),flush=True)
    imageio.imwrite(out/f'seed_{seed}_last.jpg',frames[-1])
    env.close();del context,env,agent
assert digest(checkpoint)==before
report={'purpose':'Supplementary fixed-seed final-checkpoint demo, excluded from original 8/10 evaluation',
        'checkpoint_sha256':before,'checkpoint_unchanged':True,'optimizer_updates':0,
        'seeds':[101,102,103],'episodes':rows,'native_steps':600}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('FINAL_DEMO_COMPLETE',flush=True)
