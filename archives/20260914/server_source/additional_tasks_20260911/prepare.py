"""Prepare two additional paper tasks using frozen, verified runners."""
import ast,hashlib,json,pathlib,shutil,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent
TASKS={'dmc':dict(original='dmc_tdmpc2_original',folder='walker_run',task='walker-run',gpu=0,envvar='DMC_ROOT',envhash='cba67740'),
       'ms2':dict(original='maniskill2_tdmpc2_original',folder='lift_cube',task='lift-cube',gpu=1,envvar='MS_ROOT',envhash='eff3d0dd')}
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
manifest={}
for tag,t in TASKS.items():
 src=BASE/t['original'];dst=ROOT/t['folder'];assert not dst.exists(),dst
 spec=json.loads((src/'configs/env-spec.json').read_text())
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==t['envhash']
 for folder in ['configs','artifacts','logs','reports','reference']:(dst/folder).mkdir(parents=True,exist_ok=True)
 plan=json.loads((src/'configs/plan.json').read_text());plan['task']=t['task']
 plan['reference_csv']=f"https://raw.githubusercontent.com/nicklashansen/tdmpc2/57158282b46ebc5c329c5be9cfe2b0094126d1ca/results/{t['task']}.csv"
 plan['execution']='3seeds sequentially per task; at most2training processes per GPU across paired tasks'
 if tag=='ms2':plan['env_id']='LiftCube-v0'
 save(dst/'configs/plan.json',plan);shutil.copyfile(src/'configs/env-spec.json',dst/'configs/env-spec.json')
 code=(src/'runner.py').read_text();code=once(code,"UPSTREAM = ROOT.parent / 'tdmpc2_original'",f"UPSTREAM = Path({str(BASE/'tdmpc2_original')!r})")
 if tag=='dmc':
  code=code.replace('walker_walk_seed','walker_run_seed')
  code=once(code,"suite.load('walker', 'walk',", "suite.load('walker', 'run',")
 else:
  code=code.replace('pick_cube_seed','lift_cube_seed')
  code=once(code,'assert obs.shape == (51,)','assert obs.shape == (42,)')
  code=once(code,'tuple(c.obs_shape.state) == (51,)','tuple(c.obs_shape.state) == (42,)')
  code=once(code,'native.obj.set_pose(Pose(native.goal_pos))','native.obj.set_pose(Pose([0, 0, 0.3]))')
  code=once(code,'native.obj.set_pose(Pose(native.goal_pos + np.array([0.1,0,0])))','native.obj.set_pose(Pose([0, 0, 0.02]))')
 ast.parse(code);(dst/'runner.py').write_text(code)
 runtime=f'''#!/usr/bin/env bash
set -euo pipefail
source {src}/runtime.sh
export {t['envvar']}={dst}
export XDG_CACHE_HOME={dst}/cache/xdg TORCH_HOME={dst}/cache/torch TMPDIR={dst}/tmp
export MPLCONFIGDIR={dst}/cache/matplotlib CUDA_CACHE_PATH={dst}/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd {dst}
'''
 (dst/'runtime.sh').write_text(runtime)
 (dst/'run.sh').write_text(f'#!/usr/bin/env bash\nset -euo pipefail\nexec bash {ROOT}/shared_run.sh {tag} new "$@"\n')
 final=(src/'finalize.py').read_text()
 for a,b in ([('walker_walk','walker_run'),('walker-walk','walker-run'),('Walker Walk','Walker Run')] if tag=='dmc' else [('pick_cube','lift_cube'),('pick-cube','lift-cube'),('Pick Cube','Lift Cube')]):final=final.replace(a,b)
 ast.parse(final);(dst/'finalize.py').write_text(final)
 with urllib.request.urlopen(plan['reference_csv'],timeout=40) as response:(dst/'reference'/f"{t['task']}.csv").write_bytes(response.read())
 manifest[tag]={**t,'original_root':str(src),'new_root':str(dst),'original_runner_sha256':hashlib.sha256((src/'runner.py').read_bytes()).hexdigest(),
  'new_runner_sha256':hashlib.sha256((dst/'runner.py').read_bytes()).hexdigest()}
save(ROOT/'manifest.json',manifest)
print('ADDITIONAL_TASKS_PREPARED',json.dumps(manifest))
