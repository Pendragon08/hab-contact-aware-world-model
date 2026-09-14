"""Adopt all existing training without restart, then run six independent task streams."""
import concurrent.futures,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
PREV=ROOT.parent/'additional_tasks_20260911'
REG=json.loads((ROOT/'registry.json').read_text())
LOCK=threading.RLock();STATE={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'streams':{}}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,x):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def update(task,**x):
 with LOCK:STATE['streams'].setdefault(task,{}).update(x,updated_utc=utc());save('state.json',STATE)
def proc(pid):
 try:
  p=pathlib.Path('/proc')/str(pid);raw=(p/'stat').read_text();s=raw[raw.rindex(')')+2:].split()
  return dict(pid=pid,state=s[0],ppid=int(s[1]),start_ticks=s[19],uid=p.stat().st_uid,argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x])
 except (FileNotFoundError,ProcessLookupError):return None
def live(identity):
 s=proc(identity['pid']);return s is not None and s['start_ticks']==identity['start_ticks'] and s['state']!='Z'
def verify(task):
 job=pathlib.Path(REG[task]['root'])
 for n,h in REG[task]['frozen'].items():assert hashlib.sha256((job/n).read_bytes()).hexdigest()==h,(task,n)
def artifact(task,seed):return pathlib.Path(REG[task]['root'])/'artifacts'/f'{REG[task]["stem"]}_seed{seed}_1m'
def complete(task,seed):
 p=artifact(task,seed)/'complete.json'
 if not p.exists():return False
 c=json.loads(p.read_text());assert c['completed'] and c['train_environment_steps']==1000000 and c['optimizer_updates']==499999
 return True
def archive(task,seed):
 assert complete(task,seed);verify(task)
 job=pathlib.Path(REG[task]['root']);d=artifact(task,seed)
 b=job/'release'/f'{d.name}.tar.gz';m=job/'release'/f'{d.name}.manifest.json'
 if b.exists() and m.exists():return
 assert not b.exists() and not m.exists(),'Partial archive; preserve for investigation'
 with (ROOT/'logs'/f'{task}_archive_seed{seed}.log').open('x') as f:subprocess.run([sys.executable,str(job/'finalize.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
def retire_parent(identity):
 if identity and live(identity):
  assert identity['uid']==os.getuid()
  os.kill(identity['pid'],signal.SIGTERM)
  try:os.kill(identity['pid'],signal.SIGCONT)
  except ProcessLookupError:pass
def spawn(task,seed):
 verify(task);d=artifact(task,seed);assert not d.exists(),d
 log=pathlib.Path(REG[task]['root'])/'logs'/f'seed{seed}_1m.log'
 with log.open('x') as f:p=subprocess.Popen(['bash',str(ROOT/'run.sh'),task,'train','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 update(task,status='training',seed=seed,pid=p.pid,artifact=str(d),log=str(log))
 assert p.wait()==0,('Training failed',task,seed)
 archive(task,seed);update(task,status='seed_completed',seed=seed,pid=None)
def stream(task,h):
 first=1
 if h:
  seed=h['seed'];update(task,status='adopted_training',seed=seed,pid=h['child']['pid'],artifact=str(artifact(task,seed)))
  while live(h['child']):time.sleep(5)
  retire_parent(h.get('parent'))
  assert complete(task,seed),('Adopted training exited incomplete',task,seed)
  archive(task,seed);update(task,status='seed_completed',seed=seed,pid=None);first=seed+1
 for seed in range(first,4):spawn(task,seed)
 update(task,status='all_complete',pid=None)
def guarded(task,h):
 try:stream(task,h)
 except BaseException:update(task,status='failed',error=traceback.format_exc());print('STREAM_FAILED',task,traceback.format_exc(),flush=True)
def take_handoff():
 assert not (ROOT/'handoff.json').exists(),'Already handed off'
 candidates=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:s=proc(int(p.name))
  except PermissionError:continue
  if s and s['uid']==os.getuid() and str(PREV/'manager.py') in s['argv']:candidates.append(s)
 assert len(candidates)==1,candidates
 old=candidates[0];original=json.loads((PREV/'handoff.json').read_text())['handoffs']
 for attempt in range(120):
  assert live(old);os.kill(old['pid'],signal.SIGSTOP)
  try:
   states=json.loads((PREV/'state.json').read_text())['streams'];handoffs={};ready=True
   for task,m in REG.items():
    if 'previous' not in m:continue
    s=states[m['previous']]
    if s['status'] not in ['training','existing_training'] or not s.get('pid'):ready=False;break
    child=proc(s['pid'])
    if not child or child['state']=='Z':ready=False;break
    assert child['uid']==os.getuid() and ('train' in child['argv'])
    direct=str(pathlib.Path(m['root'])/'runner.py') in child['argv']
    tag,role=m['previous'].split('_')
    shared=str(PREV/'shared_entry.py') in child['argv'] and tag in child['argv'] and role in child['argv']
    assert direct or shared,(task,child)
    verify(task)
    h=dict(seed=s['seed'],child=child,parent=None)
    prior=original.get(tag)
    if role=='old' and prior and prior['child']['pid']==child['pid'] and prior['child']['start_ticks']==child['start_ticks']:h['parent']=prior['parent']
    handoffs[task]=h
   if not ready:
    os.kill(old['pid'],signal.SIGCONT);time.sleep(5);continue
   assert all(live(h['child']) for h in handoffs.values())
   payload=dict(utc=utc(),previous_manager=old,previous_state=states,handoffs=handoffs)
   save('handoff-prepared.json',payload)
   # Signal only the scheduling PID; every training child remains untouched.
   os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT)
   for _ in range(50):
    if not live(old):break
    time.sleep(.1)
   assert not live(old),'Prior manager did not exit'
   save('handoff.json',payload)
   return handoffs
  except BaseException:
   if live(old):os.kill(old['pid'],signal.SIGCONT)
   raise
 raise RuntimeError('Could not capture stable running-stream handoff')
def main():
 (ROOT/'logs').mkdir(exist_ok=True)
 report=json.loads((ROOT/'independent-validation.json').read_text());assert report['passed']
 for n,h in report['entry_hashes'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 for task in REG:verify(task)
 for task,path in report['complete_paths'].items():
  c=json.loads(pathlib.Path(path).read_text());assert c['completed'] and c['task']==task
 handoffs=take_handoff();print('SIX_TASK_HANDOFF_COMPLETE',utc(),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  futures=[pool.submit(guarded,task,handoffs.get(task)) for task in REG]
  for f in concurrent.futures.as_completed(futures):f.result()
 with LOCK:STATE.update(finished_utc=utc(),all_complete=all(s['status']=='all_complete' for s in STATE['streams'].values()));save('state.json',STATE)
if __name__=='__main__':
 try:main()
 except BaseException:save('manager-failure.json',dict(utc=utc(),error=traceback.format_exc()));raise
