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
recovery=[]
for base in (root/'artifacts').glob('bc_v*'):
    for path in base.glob('eval_v2_*_provenance.json'):
        data=json.loads(path.read_text())
        matches=candidates.get(('bc_eval_v2.py',data['source_sha256']),[])
        checkpoint=base/'student/checkpoint_05000.pt'
        plans=[p.relative_to(root).as_posix() for p in (root/'configs').glob('bc*.json') if sha(p)==data['plan_sha256']]
        row={'record':path.relative_to(root).as_posix(),'source_matches':matches,'plan_matches':plans,
             'checkpoint_sha256':sha(checkpoint),'checkpoint_matches':sha(checkpoint)==data['checkpoint_sha256']}
        recovery.append(row)
        if not matches or not plans or not row['checkpoint_matches']:missing.append(row)
out=root/'reports/bc_source_audit.json'
out.write_text(json.dumps({'passed':not missing,'source_checks':len(rows),'recovery_checks':recovery,'missing':missing,'checks':rows},indent=2))
print('BC_SOURCE_AUDIT',len(rows),'missing',len(missing));assert not missing,missing
