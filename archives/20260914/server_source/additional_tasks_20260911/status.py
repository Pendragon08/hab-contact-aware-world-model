"""Inspect all four task streams without changing training state."""
import argparse,datetime,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--save',action='store_true');args=p.parse_args()
manifest=json.loads((ROOT/'manifest.json').read_text());state=json.loads((ROOT/'state.json').read_text())
out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tasks':[],'manager_state':state}
for tag,m in manifest.items():
 for role in ['old','new']:
  root=pathlib.Path(m['original_root'] if role=='old' else m['new_root'])
  task=('walker-walk' if tag=='dmc' else 'pick-cube') if role=='old' else m['task']
  item={'task':task,'gpu':m['gpu'],'root':str(root),'seeds':[]}
  for d in sorted((root/'artifacts').glob('*seed*_1m')):
   r={'name':d.name,'completed':False}
   q=d/'complete.json'
   if q.exists():
    c=json.loads(q.read_text());r.update(completed=c['completed'],train_environment_steps=c['train_environment_steps'],final_eval=c['final_eval'])
   q=d/'updates.jsonl'
   if q.exists():
    lines=q.read_text().splitlines()
    try:r['latest_update']=json.loads(lines[-1])
    except json.JSONDecodeError:r['latest_update']=json.loads(lines[-2])
   q=d/'metrics.jsonl'
   if q.exists():
    es=[]
    for line in q.read_text().splitlines():
     try:x=json.loads(line)
     except json.JSONDecodeError:continue
     if x['category']=='eval':es.append(x)
    if es:r['latest_eval']=es[-1]
   item['seeds'].append(r)
  out['tasks'].append(item)
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
out['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
if args.save:(ROOT/'latest-status.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
