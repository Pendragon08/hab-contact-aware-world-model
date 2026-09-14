"""Read-only wall-clock throughput sampler. Does not load ML packages."""
import argparse,datetime,json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
BASE=ROOT.parent
def snapshot():
 reg=json.loads((ROOT/'registry.json').read_text()) if (ROOT/'registry.json').exists() else {
  'walker-walk':dict(root=str(BASE/'dmc_tdmpc2_original'),stem='walker_walk',gpu=0),
  'walker-run':dict(root=str(BASE/'additional_tasks_20260911/walker_run'),stem='walker_run',gpu=0),
  'pick-cube':dict(root=str(BASE/'maniskill2_tdmpc2_original'),stem='pick_cube',gpu=1),
  'lift-cube':dict(root=str(BASE/'additional_tasks_20260911/lift_cube'),stem='lift_cube',gpu=1)}
 tasks={}
 for name,m in reg.items():
  seeds=[]
  for seed in [1,2,3]:
   d=pathlib.Path(m['root'])/'artifacts'/f'{m["stem"]}_seed{seed}_1m'
   if not d.exists():continue
   rows=(d/'updates.jsonl').read_text().splitlines() if (d/'updates.jsonl').exists() else []
   latest={}
   for line in reversed(rows):
    try:latest=json.loads(line);break
    except json.JSONDecodeError:continue
   c=json.loads((d/'complete.json').read_text()) if (d/'complete.json').exists() else {}
   seeds.append(dict(seed=seed,complete=bool(c.get('completed')),steps=c.get('train_environment_steps',latest.get('train_environment_steps',0)),updates=c.get('optimizer_updates',latest.get('update',0)),latest=latest))
  tasks[name]=dict(gpu=m['gpu'],seeds=seeds,total_steps=sum(s['steps'] for s in seeds))
 return dict(time=time.time(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=tasks,
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True),
  compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--seconds',type=int,default=300);a=p.parse_args()
 out=ROOT/'measurements';out.mkdir(exist_ok=True);start=time.monotonic()
 with (out/f'{a.label}.jsonl').open('x') as f:
  while True:
   row=snapshot();f.write(json.dumps(row)+'\n');f.flush()
   if time.monotonic()-start>=a.seconds:break
   time.sleep(10)
 print('SAMPLE_COMPLETE',a.label,flush=True)
