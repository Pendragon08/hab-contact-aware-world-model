"""Match the files actually recorded at launch to the final source tree."""
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['HAB_ROOT']);checks=[]
provenances=list((root/'artifacts/contact_v1').glob('provenance_*.json'))+list((root/'artifacts').glob('state_*_v1/provenance.json'))
assert len(provenances)>=5
for path in provenances:
    record=json.loads(path.read_text())
    for name,expected in record['source_hashes'].items():
        source=root/name
        assert source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==expected,(path,name)
        checks.append({'run':str(path.parent.relative_to(root)),'source':name,'sha256':expected})
out=root/'reports/source_audit.json';out.write_text(json.dumps({'passed':True,'recorded_sources_checked':len(checks),'checks':checks},indent=2))
print('RUN_SOURCE_AUDIT_PASS',len(checks))
