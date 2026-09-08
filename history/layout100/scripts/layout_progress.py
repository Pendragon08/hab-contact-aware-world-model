import json
from pathlib import Path
from collections import Counter
root=Path('/data/code/ouyanghao/oyh/maniskill-hab-demo')
rows=[json.loads(p.read_text()) for p in sorted((root/'results/pick_val_layout100_v1').glob('episode_*/episode.json'))]
print(json.dumps({'completed':len(rows),'layouts':dict(Counter(r['build_config_idx'] for r in rows)),'once':sum(r['success_once'] for r in rows),'end':sum(r['success_at_end'] for r in rows),'outcomes':dict(Counter(r['outcome_category'] for r in rows)),'completed_slots':len(list((root/'results/pick_val_layout100_v1').glob('complete_run_*.json')))}))
