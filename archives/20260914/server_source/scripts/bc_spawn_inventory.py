"""Read-only inventory of available public training spawn rows; never runs simulation."""
import json
from pathlib import Path
import torch
from mani_skill import ASSET_DIR
from bc_pipeline import ROOT

path = ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/spawn_data/tidy_house/pick/train/spawn_data.pt'
data = torch.load(path, map_location='cpu', weights_only=False)
rows = {str(k): {name: len(value) for name, value in val.items()} for k, val in data.items()}
lengths = [min(val.values()) for val in rows.values()]
result = {'task_keys': len(rows), 'min_rows': min(lengths), 'max_rows': max(lengths), 'task_rows': rows}
(ROOT/'reports/bc_spawn_inventory.json').write_text(json.dumps(result, indent=2))
print('SPAWN_INVENTORY', len(rows), min(lengths), max(lengths), flush=True)
