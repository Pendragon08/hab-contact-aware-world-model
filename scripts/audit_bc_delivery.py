"""Persist a local byte-for-byte delivery audit and copy release indexes into Git."""
import hashlib
import json
import shutil
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

root = Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

inventory_path=root/'release/bc-pilot-v1.release.json'
inventory=json.loads(inventory_path.read_text())
manifest_path=root/'release'/inventory['manifest']
assert sha(manifest_path)==inventory['manifest_sha256']
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))['files']
for part in inventory['parts']:
    p=root/'release'/part['name']; assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256']
for row in manifest:
    p=root/row['path']; assert p.stat().st_size==row['size'] and sha(p)==row['sha256'],row['path']

class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('src','href') and v and not urlsplit(v).scheme and not v.startswith('#'):
                self.local.append(unquote(urlsplit(v).path))
parser=Links();parser.feed((root/'行为克隆实验汇总.html').read_text(encoding='utf-8'))
assert all((root/p).is_file() for p in parser.local)
free=shutil.disk_usage(root).free; assert free>=2*2**30
target=root/'reports/release-index';target.mkdir(exist_ok=True)
for p in (inventory_path,manifest_path):shutil.copyfile(p,target/p.name)
result={'passed':True,'files_verified':len(manifest),'uncompressed_bytes':inventory['uncompressed_bytes'],
        'archive_bytes':inventory['archive_bytes'],'archive_sha256':inventory['archive_sha256'],
        'manifest_sha256':inventory['manifest_sha256'],'local_root':str(root),
        'html_local_links_verified':len(parser.local),'disk_free_bytes_after_restore':free,
        'reserve_bytes':2*2**30,'note':'Core experiment bundle checked per file after restore; later documentation and presentation audits are tracked separately in Git.'}
(root/'reports/bc_local_transfer_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('BC_LOCAL_DELIVERY_AUDIT_PASS',json.dumps(result,ensure_ascii=False))
