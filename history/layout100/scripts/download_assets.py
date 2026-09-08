"""Stage official assets without overwriting existing asset directories."""
import concurrent.futures, hashlib, json, os, time, zipfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

root = Path(os.environ['HAB_ROOT'])
data = root / 'assets/data'
inv = root / 'inventory'
inv.mkdir(exist_ok=True)
api = HfApi(token=False)

def extract(archive, target):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        prefixes = {i.filename.split('/')[0] for i in files}
        strip = next(iter(prefixes)) + '/' if len(prefixes) == 1 and all('/' in i.filename for i in files) else ''
        print('EXTRACT', archive, '->', target, 'strip', strip, 'files', len(files), flush=True)
        for info in files:
            name = info.filename[len(strip):] if strip else info.filename
            out = (target / name).resolve()
            if not out.is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, out.open('wb') as dst:
                import shutil
                shutil.copyfileobj(src, dst)

def task(name, repo, kind, target, filename=None, patterns=None, checksum=None):
    info = api.repo_info(repo, repo_type=kind)
    sha = info.sha
    print('START', name, sha, flush=True)
    if filename:
        p = hf_hub_download(repo, filename, repo_type=kind, revision=sha, token=False)
        digest = hashlib.file_digest(open(p, 'rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
            digest = h.hexdigest()
        if checksum and digest != checksum: raise ValueError('Checksum mismatch: ' + name)
        extract(p, target)
    else:
        snapshot_download(repo, repo_type=kind, revision=sha, local_dir=target, allow_patterns=patterns, token=False, max_workers=6)
        digest = None
    record = dict(name=name, repo=repo, repo_type=kind, revision=sha, filename=filename, sha256=digest, target=str(target), finished_unix=time.time())
    (inv / (name + '_download.json')).write_text(json.dumps(record, indent=2))
    print('DONE', name, flush=True)

jobs = [
    ('replicacad', 'haosulab/ReplicaCAD', 'dataset', data/'scene_datasets/replica_cad_dataset'),
    ('rearrange', 'haosulab/ReplicaCADRearrange', 'dataset', data/'scene_datasets/replica_cad_dataset/rearrange', 'rearrange.zip'),
    ('ycb', 'haosulab/ManiSkill2', 'dataset', data/'assets/mani_skill2_ycb', 'data/mani_skill2_ycb.zip', None, '1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3'),
    ('pick_policy', 'arth-shukla/mshab_checkpoints', 'model', data/'mshab_checkpoints', None, ['rl/tidy_house/pick/all/*']),
]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(task, *j) for j in jobs]
    for f in concurrent.futures.as_completed(futures): f.result()
print('ASSETS_COMPLETE', flush=True)
