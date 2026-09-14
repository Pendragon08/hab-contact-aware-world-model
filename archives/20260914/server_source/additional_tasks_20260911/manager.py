"""Two task streams per GPU, with lossless handoff of running old seed jobs."""
import concurrent.futures,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time,traceback
ROOT=pathlib.Path(__file__).resolve().parent
MANIFEST=json.loads((ROOT/'manifest.json').read_text())
LOCK=threading.RLock();STATE={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'streams':{}}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(name,obj):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def update(key,**data):
 with LOCK:
  STATE['streams'].setdefault(key,{}).update(data,updated_utc=utc());save('state.json',STATE)
def proc(pid):
 try:
  p=pathlib.Path('/proc')/str(pid);raw=(p/'stat').read_text();fields=raw[raw.rindex(')')+2:].split()
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
  return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'start_ticks':fields[19],'argv':argv,'uid':p.stat().st_uid}
 except (FileNotFoundError,ProcessLookupError):return None
def live(identity):
 s=proc(identity['pid']);return s is not None and s['start_ticks']==identity['start_ticks'] and s['state']!='Z'
def runpath(tag,role,seed):
 m=MANIFEST[tag];root=pathlib.Path(m['original_root'] if role=='old' else m['new_root'])
 stem=('walker_walk' if tag=='dmc' else 'pick_cube') if role=='old' else m['folder']
 return root,root/'artifacts'/f'{stem}_seed{seed}_1m'
def completed(tag,role,seed):
 root,d=runpath(tag,role,seed);p=d/'complete.json'
 if not p.exists():return False
 c=json.loads(p.read_text());assert c['completed'] and c['train_environment_steps']==1000000 and c['optimizer_updates']==499999
 return True
def archive(tag,role,seed):
 root,d=runpath(tag,role,seed)
 assert completed(tag,role,seed)
 bundle=root/'release'/f'{d.name}.tar.gz';manifest=root/'release'/f'{d.name}.manifest.json'
 if bundle.exists() and manifest.exists():return
 assert not bundle.exists() and not manifest.exists(),'Partial archive exists; do not overwrite'
 with (ROOT/'logs'/f'{tag}_{role}_archive_seed{seed}.log').open('x') as f:
  subprocess.run([sys.executable,str(root/'finalize.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
def verify_original(tag,seed):
 root,d=runpath(tag,'old',seed)
 for name,h in json.loads((d/'sources.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
def spawn_train(tag,role,seed):
 root,d=runpath(tag,role,seed);assert not d.exists(),d
 log=root/'logs'/f'seed{seed}_1m.log'
 with log.open('x') as f:
  p=subprocess.Popen(['bash',str(ROOT/'shared_run.sh'),tag,role,'train','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 update(tag+'_'+role,status='training',seed=seed,pid=p.pid,log=str(log),artifact=str(d))
 rc=p.wait();assert rc==0,(tag,role,seed,'exit',rc)
 archive(tag,role,seed)
 update(tag+'_'+role,status='seed_completed',seed=seed,pid=None)
def old_stream(tag,h):
 if h is None:
  update(tag+'_old',status='all_complete');return
 seed=h['seed'];update(tag+'_old',status='existing_training',seed=seed,pid=h['child']['pid'])
 while live(h['child']):time.sleep(10)
 # Only retire the stopped shell after its original training child has exited.
 parent=proc(h['parent']['pid'])
 if parent and parent['start_ticks']==h['parent']['start_ticks']:
  os.kill(parent['pid'],signal.SIGTERM)
  try:os.kill(parent['pid'],signal.SIGCONT)
  except ProcessLookupError:pass
 assert completed(tag,'old',seed),'Existing training did not complete; future seeds stopped'
 verify_original(tag,seed);archive(tag,'old',seed)
 update(tag+'_old',status='seed_completed',seed=seed,pid=None)
 for next_seed in range(seed+1,4):spawn_train(tag,'old',next_seed)
 update(tag+'_old',status='all_complete',pid=None)
def new_stream(tag):
 for seed in [1,2,3]:spawn_train(tag,'new',seed)
 update(tag+'_new',status='all_complete',pid=None)
def guarded(key,fn,*args):
 try:fn(*args)
 except BaseException:
  update(key,status='failed',error=traceback.format_exc());raise
def discover(tag):
 m=MANIFEST[tag];root=pathlib.Path(m['original_root'])
 if all(completed(tag,'old',s) for s in [1,2,3]):return None
 for _ in range(24):
  found=[]
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:s=proc(int(p.name))
   except PermissionError:continue
   if s and s['uid']==os.getuid() and str(root/'runner.py') in s['argv'] and 'train' in s['argv']:found.append(s)
  if found:
   assert len(found)==1,found
   child=found[0];seed=int(child['argv'][child['argv'].index('--seed')+1]);parent=proc(child['ppid'])
   q=root/('queue_1m.sh' if tag=='dmc' else 'queue.sh')
   assert parent and parent['uid']==os.getuid() and str(q) in parent['argv'],parent
   verify_original(tag,seed)
   return {'child':child,'parent':parent,'seed':seed}
  if all(completed(tag,'old',s) for s in [1,2,3]):return None
  time.sleep(5)
 raise RuntimeError('No running original training found for '+tag)
def main():
 assert not (ROOT/'handoff.json').exists(),'Manager already started'
 (ROOT/'logs').mkdir(exist_ok=True)
 report=json.loads((ROOT/'independent-validation.json').read_text());assert report['passed']
 for tag in MANIFEST:
  d=pathlib.Path(report['complete_paths'][tag]);c=json.loads(d.read_text());assert c['completed']
  for name,h in json.loads((d.parent/'sources.json').read_text()).items():assert hashlib.sha256((pathlib.Path(MANIFEST[tag]['new_root'])/name).read_bytes()).hexdigest()==h
  for name,h in report['entry_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 handoffs={tag:discover(tag) for tag in MANIFEST}
 save('handoff-prepared.json',{'utc':utc(),'handoffs':handoffs})
 stopped=[]
 try:
  for tag,h in handoffs.items():
   if h is None:continue
   assert live(h['child']) and live(h['parent'])
   os.kill(h['parent']['pid'],signal.SIGSTOP);stopped.append(h['parent']['pid'])
   assert live(h['child']),'Original child exited during handoff'
  save('handoff.json',{'utc':utc(),'handoffs':handoffs,'policy':'Only queue parents suspended; current training children preserved'})
 except BaseException:
  for pid in stopped:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass
  raise
 print('HANDOFF_COMPLETE',utc(),json.dumps({k:v['child']['pid'] if v else None for k,v in handoffs.items()}),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[]
  for tag,h in handoffs.items():
   futures.append(pool.submit(guarded,tag+'_old',old_stream,tag,h))
   futures.append(pool.submit(guarded,tag+'_new',new_stream,tag))
  for f in concurrent.futures.as_completed(futures):
   try:f.result()
   except BaseException:print('STREAM_FAILED',traceback.format_exc(),flush=True)
 with LOCK:
  STATE['finished_utc']=utc();STATE['all_complete']=all(s['status']=='all_complete' for s in STATE['streams'].values());save('state.json',STATE)
 print('ALL_STREAMS_FINISHED',utc(),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  save('manager-failure.json',{'utc':utc(),'error':traceback.format_exc()});raise
