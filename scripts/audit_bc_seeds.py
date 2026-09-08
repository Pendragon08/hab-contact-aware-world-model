"""Check explicit historical seed fields before using the frozen BC seed ledger."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
plan = json.loads((root/'configs/bc_v1.json').read_text())
new = set(plan['interface_seeds']) | set(range(plan['collection_seed_base'], plan['collection_seed_base']+100)) | {plan['training_seed'], plan['witness_seed']}
seen = set(); sources = []; conflicts = []
def visit(x, path):
    if isinstance(x, dict):
        for k, v in x.items():
            if 'seed' in k.lower():
                values = [v] if isinstance(v, int) else v if isinstance(v, list) else []
                for val in values:
                    if isinstance(val, int):
                        seen.add(val)
                        if val in new: conflicts.append({'path': path, 'key': k, 'seed': val})
            visit(v, path)
    elif isinstance(x, list):
        for v in x: visit(v, path)
for base in ['configs', 'reports', 'artifacts']:
    for p in (root/base).rglob('*.json'):
        if p.name == 'bc_v1.json' or 'bc_v1' in p.parts or p.name.startswith('bc_'): continue
        try: data = json.loads(p.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError): continue
        visit(data, p.relative_to(root).as_posix()); sources.append(p.relative_to(root).as_posix())
result = {'passed': not conflicts, 'new_unique_seeds': len(new), 'historical_explicit_seeds': len(seen),
          'historical_json_files': len(sources), 'conflicts': conflicts,
          'scope': 'explicit seed fields plus stored per-episode records; floating-point values are not seed matches'}
(root/'reports/bc_seed_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result)); assert not conflicts
