import json,pathlib,subprocess,sys,time
root=pathlib.Path(__file__).parent
reg=json.loads((root/'registry.json').read_text());streams={}
for name,r in reg.items():
 p=subprocess.Popen([sys.executable,str(pathlib.Path(r['root'])/'runner.py'),'train','--seed','1'],start_new_session=True)
 streams[r['previous']]=dict(status='training',seed=1,pid=p.pid)
(root/'state.json').write_text(json.dumps(dict(streams=streams)))
while True:time.sleep(1)
