"""One-shot dependency queue: start HAB only after Meta-World completes."""
import pathlib,json,time,subprocess,os,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];HERE=ROOT/'hab_tdmpc2_original'
validation=json.loads((HERE/'reports/agent-follows-doc.json').read_text())
assert validation['passed'] is True
def verify_validated_sources():
    for n,h in validation['source_sha256'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,('Validated source changed',n)
verify_validated_sources()
out=HERE/'artifacts/queue';out.mkdir(parents=True,exist_ok=False)
started=time.time();dependency=ROOT/'tdmpc2_original/artifacts/seed1_500k/complete.json'
def state(status,**kwargs):
    temp=out/'status.tmp'
    temp.write_text(json.dumps({'status':status,'utc_epoch':time.time(),**kwargs},indent=2))
    temp.replace(out/'status.json')
    print(status,kwargs,flush=True)
state('waiting_for_metaworld',dependency=str(dependency))
while True:
    if dependency.exists():
        try:d=json.loads(dependency.read_text())
        except json.JSONDecodeError:
            time.sleep(30);continue
        assert d['train_native_steps']==500000
        used=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
        if used<500:break
    if time.time()-started>24*3600:
        state('dependency_timeout_no_hab_started');raise SystemExit(2)
    time.sleep(30)
state('starting_hab')
verify_validated_sources()
with (HERE/'logs/train.log').open('x') as log:
    result=subprocess.run(['bash',str(HERE/'run.sh'),'train'],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    state('hab_failed',exit_code=result.returncode);raise SystemExit(result.returncode)
assert (HERE/'artifacts/seed1_500k/complete.json').exists()
state('training_complete_archiving')
with (HERE/'logs/archive.log').open('x') as log:
    result=subprocess.run([os.environ['HAB_PYTHON'],str(ROOT/'scripts/archive_artifacts.py'),'--name','hab-original-tdmpc2-seed1-500k',
        '--paths','hab_tdmpc2_original/artifacts/seed1_500k','hab_tdmpc2_original/reports','hab_tdmpc2_original/logs/train.log','--part-mib','500'],stdout=log,stderr=subprocess.STDOUT)
state('complete' if result.returncode==0 else 'archive_failed',archive_exit_code=result.returncode)
