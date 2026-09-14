"""Generate isolated task-specific instrumentation around pinned official TD-MPC2."""
import ast,hashlib,json,pathlib,shutil,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent
PREV=BASE/'additional_tasks_20260911'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
registry={
 'walker-walk':dict(gpu=0,root=str(BASE/'dmc_tdmpc2_original'),stem='walker_walk',previous='dmc_old'),
 'walker-run':dict(gpu=0,root=str(PREV/'walker_run'),stem='walker_run',previous='dmc_new'),
 'pick-cube':dict(gpu=1,root=str(BASE/'maniskill2_tdmpc2_original'),stem='pick_cube',previous='ms2_old'),
 'lift-cube':dict(gpu=1,root=str(PREV/'lift_cube'),stem='lift_cube',previous='ms2_new'),
}
for task,source,tag,obs in [('cheetah-run','walker_run','dmc',17),('stack-cube','lift_cube','ms2',55)]:
 src=PREV/source;dst=ROOT/task.replace('-','_');assert not dst.exists()
 for sub in ['configs','artifacts','logs','reports','reference']:(dst/sub).mkdir(parents=True)
 spec=json.loads((src/'configs/env-spec.json').read_text())
 envhash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
 assert envhash==('cba67740' if tag=='dmc' else 'eff3d0dd')
 shutil.copyfile(src/'configs/env-spec.json',dst/'configs/env-spec.json')
 plan=json.loads((src/'configs/plan.json').read_text());plan['task']=task
 plan['execution']='3seeds sequentially; at most3training processes per GPU'
 plan['reference_csv']=f'https://raw.githubusercontent.com/nicklashansen/tdmpc2/57158282b46ebc5c329c5be9cfe2b0094126d1ca/results/{task}.csv'
 if tag=='ms2':plan['env_id']='StackCube-v0'
 save(dst/'configs/plan.json',plan)
 code=(src/'runner.py').read_text().replace(source+'_seed',task.replace('-','_')+'_seed')
 if tag=='dmc':
  code=once(code,"suite.load('walker', 'run',","suite.load('cheetah', 'run',")
  code=once(code,'tuple(c.obs_shape.state) == (24,)','tuple(c.obs_shape.state) == (17,)')
 else:
  code=code.replace('Original ManiSkill2 Pick Cube','Original ManiSkill2 Stack Cube')
  code=once(code,'assert obs.shape == (42,)','assert obs.shape == (55,)')
  code=once(code,'tuple(c.obs_shape.state) == (42,)','tuple(c.obs_shape.state) == (55,)')
  code=once(code,"    initial = native.evaluate()\n    native.obj.set_pose(Pose([0, 0, 0.3]))", "    # Reset contact records before constructing a separate synthetic state.\n    env.reset()\n    initial = native.evaluate()\n    native.cubeB.set_pose(Pose([0.15, 0.15, 0.02]))\n    native.cubeA.set_pose(Pose([0.15, 0.15, 0.06]))\n    native.cubeA.set_velocity(np.zeros(3))\n    native.cubeA.set_angular_velocity(np.zeros(3))")
  code=once(code,'native.obj.set_pose(Pose([0, 0, 0.02]))','native.cubeA.set_pose(Pose([-0.15, -0.15, 0.02]))')
 ast.parse(code);(dst/'runner.py').write_text(code)
 original=BASE/('dmc_tdmpc2_original' if tag=='dmc' else 'maniskill2_tdmpc2_original')
 envvar='DMC_ROOT' if tag=='dmc' else 'MS_ROOT'
 (dst/'runtime.sh').write_text(f'''#!/usr/bin/env bash
set -euo pipefail
source {original}/runtime.sh
export {envvar}={dst}
export XDG_CACHE_HOME={dst}/cache/xdg TORCH_HOME={dst}/cache/torch TMPDIR={dst}/tmp
export MPLCONFIGDIR={dst}/cache/matplotlib CUDA_CACHE_PATH={dst}/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd {dst}
''')
 (dst/'run.sh').write_text(f'#!/usr/bin/env bash\nset -euo pipefail\nexec bash {ROOT}/run.sh {task} "$@"\n')
 final=(src/'finalize.py').read_text()
 for a,b in [(source,task.replace('-','_')),(source.replace('_','-'),task),(source.replace('_',' ').title(),task.replace('-',' ').title())]:final=final.replace(a,b)
 ast.parse(final);(dst/'finalize.py').write_text(final)
 with urllib.request.urlopen(plan['reference_csv'],timeout=40) as r:(dst/'reference'/f'{task}.csv').write_bytes(r.read())
 registry[task]=dict(gpu=0 if tag=='dmc' else 1,root=str(dst),stem=task.replace('-','_'),envhash=envhash,obs=obs)
for task,m in registry.items():
 job=pathlib.Path(m['root'])
 m['frozen']={n:sha(job/n) for n in ['runner.py','runtime.sh','run.sh','configs/plan.json','configs/env-spec.json']}
save(ROOT/'registry.json',registry)
print('SIX_TASKS_PREPARED',json.dumps(registry))
