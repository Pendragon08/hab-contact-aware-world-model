import os, sys, json, tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'local_deps'))
os.environ['HF_HOME']=str(root/'cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import snapshot_download
meta=json.loads((root/'inventory/replicacad_official_metadata.json').read_text())
missing=json.loads((root/'inventory/replicacad_missing.json').read_text())
target=root/'staging/replicacad_missing'
snapshot_download('haosulab/ReplicaCAD',repo_type='dataset',revision=meta['sha'],allow_patterns=missing,local_dir=target,max_workers=4,token=False)
with tarfile.open(root/'replicacad_remaining.tar','w') as tar:
    for name in missing: tar.add(target/name,arcname=name)
print('STAGED_COMPLETE',len(missing),(root/'replicacad_remaining.tar').stat().st_size,flush=True)
