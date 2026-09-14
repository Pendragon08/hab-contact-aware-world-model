"""Verify or restore the 2026-09-14 complete server snapshot, using only Python stdlib.

Usage: python scripts/restore_server_archive.py --archive-dir DOWNLOADS --verify
       python scripts/restore_server_archive.py --archive-dir DOWNLOADS --restore NEW_EMPTY_DIRECTORY
Restore Linux environments on Linux. Package caches are intentionally excluded.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self, size=-1):
        if size<0:raise ValueError('Only bounded streaming reads are supported')
        chunks=[];remaining=size
        while remaining:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            b=self.current.read(remaining)
            if b:chunks.append(b);remaining-=len(b)
            else:self.current.close();self.current=None
        return b''.join(chunks)
    def close(self):
        if self.current:self.current.close()
        super().close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--verify',action='store_true');mode.add_argument('--restore',type=Path)
    a=p.parse_args();index=json.loads((a.archive_dir/'archive_index.json').read_text());manifest=json.loads((a.archive_dir/'source_manifest.json').read_text(encoding='utf-8'))
    paths=[];full=hashlib.sha256()
    for part in index['parts']:
        path=a.archive_dir/part['name'];assert path.is_file() and path.stat().st_size==part['bytes'],path
        h=hashlib.sha256()
        with path.open('rb') as f:
            while True:
                b=f.read(8*1024*1024)
                if not b:break
                h.update(b);full.update(b)
        assert h.hexdigest()==part['sha256'],path
        paths.append(path)
    assert full.hexdigest()==index['compressed_sha256']
    expected={r['path']:r for r in manifest['files']};seen=set();deferred=[]
    dest=None
    if a.restore:
        dest=a.restore.resolve()
        assert not dest.exists() or (dest.is_dir() and not any(dest.iterdir())), 'Restore directory must be empty.'
        dest.mkdir(parents=True,exist_ok=True)
    with PartsReader(paths) as reader, tarfile.open(fileobj=reader,mode='r|gz') as tar:
        for member in tar:
            name=member.name;relative=PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and name not in seen,name
            assert name in expected,name
            row=expected[name];seen.add(name)
            if member.issym():
                assert row['kind']=='symlink' and member.linkname==row['target'],name
                if dest:
                    link=dest/Path(*relative.parts)
                    target=(link.parent/member.linkname).resolve()
                    assert os.path.commonpath([str(dest),str(target)])==str(dest)
                    deferred.append((link,member.linkname))
            else:
                assert member.isfile() and member.size==row['bytes'],name
                h=hashlib.sha256();target=None
                if dest:
                    output=dest/Path(*relative.parts);output.parent.mkdir(parents=True,exist_ok=True);target=output.open('xb')
                f=tar.extractfile(member)
                while True:
                    b=f.read(4*1024*1024)
                    if not b:break
                    h.update(b)
                    if target:target.write(b)
                if target:
                    target.close();os.chmod(output,member.mode)
                assert h.hexdigest()==row['sha256'],name
            if len(seen)%10000==0:print('VERIFIED_FILES',len(seen),flush=True)
    assert seen==set(expected),(len(seen),len(expected))
    for link,target in deferred:
        link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target)
    result={'verified_files':len(seen),'regular_bytes':manifest['regular_bytes'],'compressed_sha256':index['compressed_sha256'],'restore_destination':str(dest) if dest else None,'status':'PASS'}
    (a.archive_dir/'local_restore_verification.json').write_text(json.dumps(result,indent=2))
    print('ARCHIVE_VERIFIED',json.dumps(result),flush=True)

if __name__=='__main__':main()
