"""Check a fresh GitHub checkout against all Git/Release overlapping raw bytes."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[1];checkout=args.checkout.resolve()
manifest=json.loads((root/'release/bc-pilot-v1.manifest.json').read_text(encoding='utf-8'))['files']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
tracked=set(subprocess.check_output(['git','-C',str(checkout),'ls-files','-z']).decode().strip('\0').split('\0'))
rows=[]
for row in manifest:
    if row['path'] in tracked:
        assert sha(checkout/row['path'])==row['sha256'],row['path']
        rows.append({'path':row['path'],'sha256':row['sha256']})
assert rows and 'reports/bc_v1_2/summary.json' in {r['path'] for r in rows}
for name in ['bc-pilot-v1.manifest.json','bc-pilot-v1.release.json']:
    assert sha(checkout/'reports/release-index'/name)==sha(root/'release'/name)
config=checkout/'configs/bc_v1_2.json'
assert sha(config)=='7c2dc173dae9ecd06eda6a95d3cbe27079879c045a166993d163469cbb4483a6'
commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
result={'passed':True,'fresh_clone_commit':commit,'checkout':str(checkout),'overlap_files_verified':len(rows),
        'checks':rows,'config_sha256':sha(config),
        'scope':'Fresh GitHub code/metadata checkout and every Git/Release overlapping file verified. Full raw bundle restoration was separately verified at the main D drive destination.'}
(root/'reports/bc_fresh_clone_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('BC_FRESH_GIT_OVERLAP_PASS',len(rows),commit)
