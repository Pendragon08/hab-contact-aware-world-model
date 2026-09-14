"""GPU co-residency launch policy; no model, loss or optimizer changes."""
import argparse,hashlib,importlib.util,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('tag',choices=['dmc','ms2']);p.add_argument('role',choices=['old','new']);p.add_argument('mode',choices=['train','witness']);p.add_argument('--seed',type=int,choices=[1,2,3],default=1)
a=p.parse_args();m=json.loads((ROOT/'manifest.json').read_text())[a.tag]
job=pathlib.Path(m['original_root'] if a.role=='old' else m['new_root'])
source=job/'runner.py';spec=importlib.util.spec_from_file_location('job_runner',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
original_guard=mod.resource_guard
def guard(require_idle=False):
 evidence=original_guard(False)
 if require_idle:
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
  co=[]
  for line in apps.splitlines():
   uuid,pid=[x.strip() for x in line.split(',')]
   if uuid!=evidence['gpu_uuid'] or int(pid)==os.getpid():continue
   proc=pathlib.Path('/proc')/pid;assert proc.stat().st_uid==os.getuid()
   argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   is_original=str(pathlib.Path(m['original_root'])/'runner.py') in argv and 'train' in argv
   is_shared=str(ROOT/'shared_entry.py') in argv and a.tag in argv and ('old' if a.role=='new' else 'new') in argv
   assert is_original or is_shared,('Unexpected GPU co-resident',pid,argv)
   co.append({'pid':int(pid),'argv':argv})
  assert len(co)<=1,('At most2training jobs per GPU',co)
  evidence['authorized_co_residents']=co
 return evidence
mod.resource_guard=guard
frozen={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['shared_entry.py','shared_run.sh','manifest.json']}
(ROOT/'provenance').mkdir(exist_ok=True)
record=ROOT/'provenance'/f'{a.tag}_{a.role}_{a.mode}_seed{a.seed}_{os.getpid()}.json'
record.write_text(json.dumps({'pid':os.getpid(),'tag':a.tag,'role':a.role,'mode':a.mode,'seed':a.seed,'time':time.time(),'entry_hashes':frozen,'job_root':str(job)},indent=2)+'\n')
mod.main(a.mode,a.seed)
for n,h in frozen.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
