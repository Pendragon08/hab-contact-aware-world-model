import hashlib,tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
archive=root/'staging/layout100_results_v1.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='93230e7ebef0d9984d5083657189a38e2bf5ff0cfd5a772b6f9200edf7714910'
with tarfile.open(archive,'r:gz',encoding='utf-8') as tar:
    members=tar.getmembers()
    for member in members:
        target=(root/member.name).resolve()
        assert target.is_relative_to(root.resolve()),member.name
        assert member.isfile() or member.isdir(),member.name
    tar.extractall(root,members=members,filter='data')
print('UTF8_ARCHIVE_EXTRACTED',len(members))
