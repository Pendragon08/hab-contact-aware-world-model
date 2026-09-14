"""Verify Release part checksums, safely extract and verify every raw file."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    info=json.loads(a.inventory.read_text());directory=a.inventory.parent
    for part in info['parts']:
        path=directory/part['name'];assert path.stat().st_size==part['bytes'] and sha(path)==part['sha256'],path
    archive=directory/info['archive']
    if len(info['parts'])>1:
        if not archive.exists():
            with archive.open('xb') as out:
                for part in info['parts']:
                    with (directory/part['name']).open('rb') as inp:shutil.copyfileobj(inp,out)
    assert sha(archive)==info['archive_sha256']
    manifest_path=directory/info['manifest'];assert sha(manifest_path)==info['manifest_sha256']
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))['files'];expected={row['path']:row for row in manifest}
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz',encoding='utf-8') as tar:
        members=tar.getmembers()
        assert len(members)==len(expected) and {m.name for m in members}==set(expected)
        for member in members:
            target=(root/member.name).resolve()
            assert target.is_relative_to(root) and member.isfile(),member.name
            assert member.size==expected[member.name]['size']
            if target.exists():
                assert sha(target)==expected[member.name]['sha256'],f'Refuse overwrite: {target}'
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
            assert sha(target)==expected[member.name]['sha256'],target
    print('RESTORE_VERIFIED',len(manifest),'files',flush=True)

if __name__=='__main__':main()
