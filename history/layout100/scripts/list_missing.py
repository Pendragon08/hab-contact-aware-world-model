import json, os
from pathlib import Path
root=Path(os.environ['HAB_ROOT']); meta=json.loads((root/'inventory/replicacad_official_metadata.json').read_text())
target=root/'assets/data/scene_datasets/replica_cad_dataset'
missing=[f['rfilename'] for f in meta['siblings'] if not (target/f['rfilename']).is_file() or (target/f['rfilename']).stat().st_size!=f['size']]
(root/'inventory/replicacad_missing.json').write_text(json.dumps(missing))
print('MISSING',len(missing),sum(f['size'] for f in meta['siblings'] if f['rfilename'] in missing))
