"""Create isolated bounded copies; never edit the original experiments."""
import ast,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent
JOBS=[('dmc','dmc_tdmpc2_original','DMC_ROOT',1789362,'cba67740'),
      ('ms2','maniskill2_tdmpc2_original','MS_ROOT',1795158,'eff3d0dd')]
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
manifest={}
for tag,original,envvar,pid,envhash in JOBS:
 src=BASE/original;dst=ROOT/tag
 assert not dst.exists(),dst
 spec=json.loads((src/'configs/env-spec.json').read_text())
 assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==envhash
 (dst/'configs').mkdir(parents=True)
 (dst/'artifacts').mkdir()
 plan=json.loads((src/'configs/plan.json').read_text())
 plan.update(decisions_per_seed=12000,environment_steps_per_seed=24000,seeds=[2],
             scope='bounded throughput pilot; stop at full episode after measurement; not formal result')
 save(dst/'configs/plan.json',plan)
 shutil.copyfile(src/'configs/env-spec.json',dst/'configs/env-spec.json')
 code=(src/'runner.py').read_text()
 code=once(code,"UPSTREAM = ROOT.parent / 'tdmpc2_original'",f"UPSTREAM = Path({str(BASE/'tdmpc2_original')!r})")
 start=code.index('    if require_idle:')
 end=code.index("    return {'gpu_uuid'",start)
 code=code[:start]+f'''    if require_idle:
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        seen = {{int(line.split(',')[1]) for line in apps.splitlines() if line.split(',')[0].strip()==uuid}}
        assert seen == {{{pid}}}, ('Unexpected co-resident compute processes',seen)
        protected = Path('/proc/{pid}/cmdline').read_bytes().replace(b'\\0',b' ').decode()
        assert {str(src/'runner.py')!r} in protected and 'train --seed 1' in protected, protected
        assert Path('/proc/{pid}').stat().st_uid == os.getuid()
'''+code[end:]
 # Preserve official buffer capacity despite the shorter pilot stopping budget.
 code=once(code,'    buffer = Buffer(c)',"    buffer_cfg = OmegaConf.create(OmegaConf.to_container(c))\n    buffer_cfg.steps = 500000\n    buffer = Buffer(buffer_cfg)")
 code=once(code,"namespace = {}",'''source = source.replace('while self._step < self.cfg.steps:',
    "while self._step < self.cfg.steps:\\n\\t\\tif self._step >= 3500 and self._step % self.cfg.episode_length == 0 and (ROOT / 'stop-after-episode').exists():\\n\\t\\t\\tself.cfg.steps = self._step\\n\\t\\t\\tbreak")
namespace = {}''')
 # Dynamically compiled official method needs the pilot root for its stop flag.
 code=once(code,"exec(compile(source, 'official_train_exact_budget', 'exec'), OnlineTrainer.train.__globals__, namespace)","train_globals = dict(OnlineTrainer.train.__globals__, ROOT=ROOT)\nexec(compile(source, 'official_train_exact_budget', 'exec'), train_globals, namespace)")
 a=code.index("    label = f'witness_");b=code.index('\n',a)
 code=code[:a]+"    label = 'throughput_seed2'"+code[b:]
 code=once(code,'        trainer.train()',"        trainer.train()\n        OmegaConf.save(c, out / 'effective-config.yaml')")
 ast.parse(code)
 (dst/'runner.py').write_text(code)
 runtime=f'''#!/usr/bin/env bash
set -euo pipefail
source {src}/runtime.sh
export {envvar}={dst}
export XDG_CACHE_HOME={dst}/cache/xdg TORCH_HOME={dst}/cache/torch TMPDIR={dst}/tmp
export MPLCONFIGDIR={dst}/cache/matplotlib CUDA_CACHE_PATH={dst}/cache/cuda
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
cd {dst}
'''
 (dst/'runtime.sh').write_text(runtime)
 (dst/'run.sh').write_text(f'#!/usr/bin/env bash\nset -euo pipefail\nsource {dst}/runtime.sh\nexec "$HAB_PYTHON" {dst}/runner.py train --seed 2\n')
 manifest[tag]={'original_root':str(src),'pilot_root':str(dst),'protected_pid':pid,'env_hash':envhash,
                'original_runner_sha256':hashlib.sha256((src/'runner.py').read_bytes()).hexdigest(),
                'pilot_runner_sha256':hashlib.sha256((dst/'runner.py').read_bytes()).hexdigest()}
save(ROOT/'manifest.json',manifest)
print('THROUGHPUT_PREPARED',json.dumps(manifest))
