"""Create complete per-experiment Release bundles and SHA256 manifests."""
import argparse,hashlib,json,tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def build(root,name,paths,part_mib):
    dest=root/'release';dest.mkdir(exist_ok=True)
    manifest=[];seen=set()
    for base in paths:
        p=root/base
        if not p.exists():raise FileNotFoundError(p)
        for f in sorted(p.rglob('*') if p.is_dir() else [p]):
            if not f.is_file():continue
            rel=f.relative_to(root).as_posix()
            if rel in seen:continue
            seen.add(rel);manifest.append({'path':rel,'size':f.stat().st_size,'sha256':sha(f)})
    manifest_path=dest/f'{name}.manifest.json'
    assert not manifest_path.exists(),manifest_path
    manifest_path.write_text(json.dumps({'bundle':name,'files':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    archive=dest/f'{name}.tar.gz';assert not archive.exists(),archive
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for row in manifest:tar.add(root/row['path'],arcname=row['path'],recursive=False)
    parts=[]
    if archive.stat().st_size>part_mib*2**20:
        with archive.open('rb') as src:
            index=0
            while True:
                data=src.read(part_mib*2**20)
                if not data:break
                part=dest/f'{name}.tar.gz.part{index:03d}'
                with part.open('xb') as f:f.write(data)
                parts.append(part);index+=1
    else:parts=[archive]
    inventory={'bundle':name,'files':len(manifest),'uncompressed_bytes':sum(r['size'] for r in manifest),
               'archive':archive.name,'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
               'manifest':manifest_path.name,'manifest_sha256':sha(manifest_path),
               'parts':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in parts]}
    (dest/f'{name}.release.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps(inventory),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--name',required=True)
    p.add_argument('--paths',nargs='+',required=True);p.add_argument('--part-mib',type=int,default=500);a=p.parse_args()
    build(a.root,a.name,a.paths,a.part_mib)
