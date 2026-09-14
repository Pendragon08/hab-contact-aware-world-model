"""Wait for real training on all six streams, then sample steady wall-clock speed."""
import json,pathlib,subprocess,sys,time
from sample import snapshot
ROOT=pathlib.Path(__file__).resolve().parent
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
 s=snapshot();state=json.loads((ROOT/'state.json').read_text())['streams']
 assert not any(v['status']=='failed' for v in state.values()),state
 ready=len(state)==6 and len(s['compute'].splitlines())==6
 for task,r in state.items():
  data=s['tasks'][task];seeds=[v for v in data['seeds'] if v['seed']==r['seed']]
  ready=ready and r['status'] in ['training','adopted_training'] and bool(seeds) and not seeds[0]['complete'] and seeds[0]['latest'].get('train_decisions',0)>=(3001 if data['gpu']==0 else 1201)
 if ready:
  (ROOT/'measurements/post-ready.json').write_text(json.dumps(s,indent=2)+'\n')
  subprocess.run([sys.executable,str(ROOT/'sample.py'),'six_steady','--seconds','300'],check=True)
  break
 time.sleep(10)
else:raise RuntimeError('Six-task steady readiness not reached in30minutes')
print('SIX_STEADY_MEASURED',flush=True)
