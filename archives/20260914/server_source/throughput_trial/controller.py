"""Measure original tasks alone, then with one identical-task pilot per GPU."""
import datetime,hashlib,json,os,pathlib,signal,statistics,subprocess,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def rows(p):
 if not p.exists():return []
 out=[]
 for line in p.read_text().splitlines():
  try:out.append(json.loads(line))
  except json.JSONDecodeError:pass # concurrent append may leave one partial line
 return out
paths={}
for tag,m in MANIFEST.items():
 original=pathlib.Path(m['original_root'])
 run='walker_walk_seed1_1m' if tag=='dmc' else 'pick_cube_seed1_1m'
 paths[tag+'_original']=original/'artifacts'/run
 paths[tag+'_pilot']=pathlib.Path(m['pilot_root'])/'artifacts/throughput_seed2'
def snap():
 d={'utc':utc(),'wall':time.time(),'runs':{},'load':os.getloadavg()}
 for k,p in paths.items():
  rs=rows(p/'updates.jsonl')
  if rs:d['runs'][k]=rs[-1]
 d['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()
 d['compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 return d
def active_rate(rs):
 # Regular 100-update intervals only; exclude zero-step seed pretraining and
 # intervals that cross scheduled evaluation boundaries.
 valid=[];dropped=0
 for a,b in zip(rs,rs[1:]):
  steps=b['train_environment_steps']-a['train_environment_steps'];dt=b['seconds']-a['seconds']
  if steps!=200 or dt<=0 or a['train_decisions']//50000!=b['train_decisions']//50000:
   dropped+=1;continue
  valid.append((steps,dt))
 assert valid,'No valid steady-state training intervals'
 return {'environment_steps_per_second':sum(x for x,t in valid)/sum(t for x,t in valid),
         'optimizer_updates_per_second':sum(x/2 for x,t in valid)/sum(t for x,t in valid),
         'measured_seconds':sum(t for x,t in valid),'intervals':len(valid),'excluded_intervals':dropped}
children={};streams=[]
try:
 assert not (ROOT/'started.json').exists(),'Trial already started'
 before=snap();baseline={}
 for tag in MANIFEST:
  p=paths[tag+'_original'];rs=rows(p/'updates.jsonl');last=rs[-1]['seconds']
  selected=[x for x in rs if x['seconds']>=last-300]
  assert selected[-1]['seconds']-selected[0]['seconds']>=270
  baseline[tag]=active_rate(selected)
  baseline[tag]['end_to_end_environment_steps_per_second']=(selected[-1]['train_environment_steps']-selected[0]['train_environment_steps'])/(selected[-1]['seconds']-selected[0]['seconds'])
  save('baseline-'+tag+'.json',selected)
 save('baseline.json',{'utc':utc(),'rates':baseline,'resources':before})
 for tag,m in MANIFEST.items():
  f=(ROOT/(tag+'.log')).open('x');streams.append(f)
  children[tag]=subprocess.Popen(['bash',str(pathlib.Path(m['pilot_root'])/'run.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save('started.json',{'utc':utc(),'pids':{k:p.pid for k,p in children.items()},'protected_pids':{k:m['protected_pid'] for k,m in MANIFEST.items()}})
 print('PILOTS_STARTED',utc(),flush=True)
 deadline=time.time()+1800;window_start=None;last=None
 while True:
  assert time.time()<deadline,'Trial timeout'
  for tag,p in children.items():assert p.poll() is None,('Pilot exited before measurement complete',tag,p.returncode)
  s=snap();last=s
  with (ROOT/'samples.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
  for tag,m in MANIFEST.items():
   assert pathlib.Path('/proc/'+str(m['protected_pid'])).exists(),'Protected training stopped'
  assert __import__('shutil').disk_usage(ROOT).free>=15*2**30
  for line in s['gpu'].splitlines():assert int(line.split(',')[2])>=12288
  ready=all(s['runs'].get(tag+'_pilot',{}).get('train_decisions',0)>=3501 for tag in MANIFEST)
  if window_start is None and ready:
   window_start=s;save('window-start.json',s);print('MEASUREMENT_STARTED',utc(),flush=True)
  if window_start and s['wall']-window_start['wall']>=300:
   save('window-end.json',s);break
  time.sleep(10)
 # Finish both pilots gracefully at the next full episode, save models, evaluate.
 for tag,m in MANIFEST.items():(pathlib.Path(m['pilot_root'])/'stop-after-episode').write_text(utc()+'\n')
 result={'started_utc':window_start['utc'],'ended_utc':last['utc'],'window_seconds':last['wall']-window_start['wall'],
         'baseline':baseline,'concurrent':{},'per_gpu':{},'scope':'bounded same-task seed2 throughput trial; not a formal1M result'}
 for key,p in paths.items():
  a=window_start['runs'][key];b=last['runs'][key]
  selected=[x for x in rows(p/'updates.jsonl') if a['update']<=x['update']<=b['update']]
  result['concurrent'][key]=active_rate(selected)
  result['concurrent'][key]['window_environment_steps_per_second']=(b['train_environment_steps']-a['train_environment_steps'])/result['window_seconds']
 for tag in MANIFEST:
  old=result['concurrent'][tag+'_original']['environment_steps_per_second']
  new=result['concurrent'][tag+'_pilot']['environment_steps_per_second']
  base=baseline[tag]['environment_steps_per_second']
  result['per_gpu'][tag]={'baseline_environment_steps_per_second':base,'original_concurrent_rate':old,
   'pilot_concurrent_rate':new,'combined_environment_steps_per_second':old+new,'throughput_factor':(old+new)/base,
   'original_slowdown_percent':100*(1-old/base)}
 save('measurement.json',result);print('MEASUREMENT_COMPLETE',json.dumps(result['per_gpu']),flush=True)
 while any(p.poll() is None for p in children.values()):
  assert time.time()<deadline,'Finalization timeout'
  time.sleep(10)
 for tag,p in children.items():
  assert p.returncode==0,(tag,p.returncode)
  c=json.loads((paths[tag+'_pilot']/'complete.json').read_text())
  assert c['completed'] and c['all_five_modules_changed'] and c['optimizer_updates']==c['train_decisions']-1
  result[tag+'_pilot_completion']=c
 for tag,m in MANIFEST.items():
  original=pathlib.Path(m['original_root'])
  assert hashlib.sha256((original/'runner.py').read_bytes()).hexdigest()==m['original_runner_sha256']
  for name,h in json.loads((paths[tag+'_original']/'sources.json').read_text()).items():
   assert hashlib.sha256((original/name).read_bytes()).hexdigest()==h
  assert pathlib.Path('/proc/'+str(m['protected_pid'])).exists()
 result['completed']=True;result['final_resources']=snap();save('result.json',result)
 print('THROUGHPUT_TRIAL_PASS',utc(),flush=True)
except BaseException:
 save('failure.json',{'utc':utc(),'traceback':traceback.format_exc()})
 for tag,p in children.items():
  if p.poll() is None:
   cmd=pathlib.Path('/proc/'+str(p.pid)+'/cmdline').read_bytes()
   if str(ROOT/tag).encode() in cmd:os.killpg(p.pid,signal.SIGTERM)
 raise
finally:
 for f in streams:f.close()
