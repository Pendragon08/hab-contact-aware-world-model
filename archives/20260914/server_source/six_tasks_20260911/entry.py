"""Resource-only co-residency policy for six official-task streams."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
REG=json.loads((ROOT/'registry.json').read_text())
p=argparse.ArgumentParser();p.add_argument('task',choices=REG);p.add_argument('mode',choices=['train','witness']);p.add_argument('--seed',type=int,choices=[1,2,3],default=1)
a=p.parse_args();job=pathlib.Path(REG[a.task]['root'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for n,h in REG[a.task]['frozen'].items():assert sha(job/n)==h,n
spec=importlib.util.spec_from_file_location('job_runner',job/'runner.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
original_guard=mod.resource_guard
def guard(require_idle=False):
 evidence=original_guard(False)
 if require_idle:
  co=[]
  for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
   uuid,pid=[x.strip() for x in line.split(',')]
   if uuid!=evidence['gpu_uuid'] or int(pid)==os.getpid():continue
   proc=pathlib.Path('/proc')/pid
   try:
    assert proc.stat().st_uid==os.getuid()
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   except FileNotFoundError:continue
   matched=[]
   for name,m in REG.items():
    if name==a.task or m['gpu']!=REG[a.task]['gpu']:continue
    direct=str(pathlib.Path(m['root'])/'runner.py') in argv and 'train' in argv
    new=str(ROOT/'entry.py') in argv and name in argv and 'train' in argv
    old=False
    if 'previous' in m:
     tag,role=m['previous'].split('_')
     old=str(ROOT.parent/'additional_tasks_20260911/shared_entry.py') in argv and tag in argv and role in argv and 'train' in argv
    if direct or new or old:matched.append(name)
   assert len(matched)==1,('Unexpected co-resident',pid,argv,matched)
   co.append(dict(pid=int(pid),task=matched[0],argv=argv))
  assert len(co)<=2 and len({r['task'] for r in co})==len(co),co
  evidence['authorized_co_residents']=co
 return evidence
mod.resource_guard=guard
frozen={n:sha(ROOT/n) for n in ['entry.py','run.sh','registry.json']}
(ROOT/'provenance').mkdir(exist_ok=True)
record=ROOT/'provenance'/f'{a.task}_{a.mode}_seed{a.seed}_{os.getpid()}.json'
record.write_text(json.dumps(dict(pid=os.getpid(),task=a.task,mode=a.mode,seed=a.seed,utc_epoch=time.time(),entry_hashes=frozen,job_root=str(job)),indent=2)+'\n')
mod.main(a.mode,a.seed)
for n,h in frozen.items():assert sha(ROOT/n)==h,n
