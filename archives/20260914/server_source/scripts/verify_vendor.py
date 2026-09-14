"""Compare vendored TD-MPC2 files to the committed upstream source lock."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
lock=json.loads((root/'configs/tdmpc2-source-lock.json').read_text())
for row in lock['files']:
    data=(root/row['path']).read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(data).hexdigest()==row['sha256_lf'] and row['upstream_match'],row['path']
print('UPSTREAM_CODE_VERIFIED',len(lock['files']),lock['upstream'])
