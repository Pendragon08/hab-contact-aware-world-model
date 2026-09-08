"""Compare mirror downloads against metadata captured directly from HF upstream."""
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['HAB_ROOT']); inv=root/'inventory'; checks=[]
for name in ['replicacad','rearrange','ycb','pick_policy']:
    official=json.loads((inv/f'{name}_official_metadata.json').read_text())
    downloaded=json.loads((inv/f'{name}_download.json').read_text())
    assert official['sha']==downloaded['revision'], name
    for f in official['siblings']:
        if name in ['rearrange','ycb']:
            if f['rfilename']!=downloaded['filename']: continue
            assert f['lfs']['sha256']==downloaded['sha256'], name
            checks.append({'name':name,'file':f['rfilename'],'sha256':downloaded['sha256']})
        elif name=='pick_policy':
            if not f['rfilename'].startswith('rl/tidy_house/pick/all/'): continue
            p=Path(downloaded['target'])/f['rfilename']
            assert p.stat().st_size==f['size']
            if 'lfs' in f:
                sha=hashlib.sha256(p.read_bytes()).hexdigest(); assert sha==f['lfs']['sha256']
                checks.append({'name':name,'file':f['rfilename'],'sha256':sha})
            else:
                b=p.read_bytes(); gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                assert gitsha==f['blobId']
        else:
            p=Path(downloaded['target'])/f['rfilename']
            assert p.is_file() and p.stat().st_size==f['size'], str(p)
    print('VERIFIED',name,official['sha'],flush=True)
(inv/'source_verification.json').write_text(json.dumps(checks,indent=2))
