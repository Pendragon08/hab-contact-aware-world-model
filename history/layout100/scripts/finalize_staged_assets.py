import hashlib,json,os,time
from pathlib import Path
root=Path(os.environ['HAB_ROOT']); inv=root/'inventory'
meta=json.loads((inv/'replicacad_official_metadata.json').read_text())
target=root/'assets/data/scene_datasets/replica_cad_dataset'
for f in meta['siblings']:
    p=target/f['rfilename']
    assert p.is_file() and p.stat().st_size==f['size'],str(p)
    if 'lfs' in f:
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for b in iter(lambda:stream.read(8*1024*1024),b''): h.update(b)
        assert h.hexdigest()==f['lfs']['sha256'],str(p)
    else:
        b=p.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['blobId'],str(p)
record=dict(name='replicacad',repo='haosulab/ReplicaCAD',repo_type='dataset',revision=meta['sha'],filename=None,sha256=None,target=str(target),finished_unix=time.time(),transport='partial HF mirror; missing files official HF local staging and scp',all_files_verified=True)
(inv/'replicacad_download.json').write_text(json.dumps(record,indent=2))
print('REPLICACAD_ALL_FILES_HASH_VERIFIED',len(meta['siblings']),flush=True)
