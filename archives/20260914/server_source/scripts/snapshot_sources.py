"""Preserve exact runtime source bytes separately from Git LF normalization."""
import hashlib,json,os,shutil
from pathlib import Path
root=Path(os.environ['HAB_ROOT']);dest=root/'artifacts/source_snapshot';dest.mkdir(parents=True,exist_ok=False)
rows=[]
for folder in ['scripts','configs','tests','vendor_tdmpc2']:
    for path in sorted((root/folder).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        rel=path.relative_to(root);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        rows.append({'path':rel.as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'sha256_lf':hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()})
environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','HAB_ROOT','HAB_DEP_ROOT','HAB_PYTHON','MS_ASSET_DIR','WM_SAPIEN_CACHE','OMP_NUM_THREADS','MKL_NUM_THREADS','WANDB_MODE']}
(dest/'source_manifest.json').write_text(json.dumps({'files':rows,'runtime_environment_allowlist':environment,
    'note':'Exact runtime bytes. Git normalizes text to LF; use these archived source copies when verifying raw launch hashes.'},indent=2))
print('EXACT_RUNTIME_SOURCES_SAVED',len(rows))
