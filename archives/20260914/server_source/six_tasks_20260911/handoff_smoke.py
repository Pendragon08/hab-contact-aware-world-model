"""Disposable CPU subprocess integration test of scheduler-only replacement."""
import importlib.util,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scheduler',ROOT/'manager.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
TMP=ROOT/'handoff-smoke';TMP.mkdir(exist_ok=False);prev=TMP/'previous';prev.mkdir()
original_root=m.ROOT;m.ROOT=TMP;m.PREV=prev
reg={}
for name,key in [('walker-walk','dmc_old'),('walker-run','dmc_new'),('pick-cube','ms2_old'),('lift-cube','ms2_new')]:
 job=TMP/name;job.mkdir()
 (job/'runner.py').write_text('import pathlib,time\np=pathlib.Path(__file__).parent/"counter"\ni=0\nwhile True:\n p.write_text(str(i));i+=1;time.sleep(.1)\n')
 reg[name]=dict(root=str(job),previous=key)
m.REG=reg;m.verify=lambda task:None
(prev/'handoff.json').write_text(json.dumps(dict(handoffs={})))
(prev/'registry.json').write_text(json.dumps(reg))
(prev/'manager.py').write_text('''import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).parent
reg=json.loads((root/'registry.json').read_text());streams={}
for name,r in reg.items():
 p=subprocess.Popen([sys.executable,str(pathlib.Path(r['root'])/'runner.py'),'train','--seed','1'],start_new_session=True)
 streams[r['previous']]=dict(status='training',seed=1,pid=p.pid)
(root/'state.json').write_text(json.dumps(dict(streams=streams)))
while True:time.sleep(1)
''')
p=subprocess.Popen([sys.executable,str(prev/'manager.py')],start_new_session=True)
children=[]
try:
 for _ in range(100):
  if (prev/'state.json').exists() and all((pathlib.Path(r['root'])/'counter').exists() for r in reg.values()):break
  time.sleep(.1)
 children=[v['pid'] for v in json.loads((prev/'state.json').read_text())['streams'].values()]
 before={k:int((pathlib.Path(v['root'])/'counter').read_text()) for k,v in reg.items()}
 h=m.take_handoff();assert p.wait(timeout=5)==-15
 time.sleep(.7)
 after={k:int((pathlib.Path(v['root'])/'counter').read_text()) for k,v in reg.items()}
 assert all(after[k]>before[k] and m.live(h[k]['child']) for k in h)
 result=dict(passed=True,manager_exit=-15,before=before,after=after,children=children,preserved_all_four=True)
 (TMP/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('SCHEDULER_HANDOFF_SMOKE_PASS',json.dumps(result))
finally:
 for pid in children:
  s=m.proc(pid)
  if s and any(str(pathlib.Path(v['root'])/'runner.py') in s['argv'] for v in reg.values()):subprocess.run(['kill','-TERM',str(pid)],check=False)
 if p.poll() is None:p.terminate();p.wait(timeout=5)
