"""Resolve every recorded runtime source hash against retained original source bytes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
candidates={}
for base in [root/'scripts',*(root/'artifacts').glob('bc_v*/source_snapshot')]:
    for path in base.rglob('*.py'):
        candidates.setdefault((path.name,sha(path)),[]).append(path.relative_to(root).as_posix())
rows=[];missing=[]
for base in (root/'artifacts').glob('bc_v*'):
    for path in base.glob('provenance_*.json'):
        data=json.loads(path.read_text())
        for name,digest in data['source_hashes'].items():
            matches=candidates.get((Path(name).name,digest),[])
            row={'record':path.relative_to(root).as_posix(),'name':name,'sha256':digest,'matches':matches}
            rows.append(row)
            if not matches:missing.append(row)
out=root/'reports/bc_source_audit.json'
out.write_text(json.dumps({'passed':not missing,'source_checks':len(rows),'missing':missing,'checks':rows},indent=2))
print('BC_SOURCE_AUDIT',len(rows),'missing',len(missing));assert not missing,missing
