"""Check parent-shell handoff behavior with disposable CPU-only stand-ins."""
import json,os,pathlib,signal,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
d=ROOT/'handoff-smoke';d.mkdir(exist_ok=False)
(d/'child.py').write_text("import pathlib,time,os\np=pathlib.Path(__file__).parent\n(p/'child.pid').write_text(str(os.getpid()))\nfor i in range(15):\n (p/'counter').write_text(str(i))\n time.sleep(.2)\n(p/'child.complete').write_text('done')\n")
(d/'parent.sh').write_text(f'#!/usr/bin/env bash\npython3 {d}/child.py\necho unexpected > {d}/next-seed-started\n')
p=subprocess.Popen(['bash',str(d/'parent.sh')],start_new_session=True)
try:
 deadline=time.time()+15
 while not (d/'counter').exists():
  assert time.time()<deadline;time.sleep(.05)
 assert str(d/'parent.sh').encode() in pathlib.Path('/proc',str(p.pid),'cmdline').read_bytes()
 os.kill(p.pid,signal.SIGSTOP);before=int((d/'counter').read_text());time.sleep(.8);after=int((d/'counter').read_text())
 assert after>before,'Stopping queue shell stopped its training child'
 while not (d/'child.complete').exists():
  assert time.time()<deadline;time.sleep(.1)
 os.kill(p.pid,signal.SIGTERM);os.kill(p.pid,signal.SIGCONT);rc=p.wait(timeout=10)
 assert not (d/'next-seed-started').exists(),'Retired shell advanced to next seed'
 out={'passed':True,'counter_before':before,'counter_after':after,'parent_exit':rc,'next_seed_not_started':True,'scope':'CPU-only disposable parent/child stand-ins'}
 (d/'result.json').write_text(json.dumps(out,indent=2)+'\n');print('HANDOFF_SMOKE_PASS',out)
finally:
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM);os.killpg(p.pid,signal.SIGCONT)
