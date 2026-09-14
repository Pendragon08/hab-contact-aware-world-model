"""Read-only per-GPU sampling; never modifies another process."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--pid',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:
    while True:
        try:os.kill(a.pid,0)
        except ProcessLookupError:break
        r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        f.write(json.dumps({'unix':time.time(),'gpu_rows':r.stdout.strip().splitlines(),'returncode':r.returncode})+'\n');f.flush()
        time.sleep(5)
