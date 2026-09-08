"""Compact read-only progress; no success-based early stopping."""
import json,os
from pathlib import Path
root=Path(os.environ.get('HAB_ROOT',Path(__file__).resolve().parents[1]))
result={'contact':{},'state':{}}
for method in ['sac','conservative','force_rule']:
    rows=list((root/'artifacts/contact_v1'/method).glob('episode_*/episode.json'))
    result['contact'][method]={'completed_episodes':len(rows),'expected':100}
for algo in ['tdmpc2','sac']:
    folder=root/f'artifacts/state_{algo}_pilot_v1'
    if (folder/'summary.json').exists():result['state'][algo]=json.loads((folder/'summary.json').read_text())
    else:
        log=folder/'updates.jsonl';last=None
        if log.exists():
            with log.open() as f:
                for line in f:
                    try:last=json.loads(line)
                    except json.JSONDecodeError:pass
        result['state'][algo]={'completed_train_episodes':len(list(folder.glob('train_*.json'))),
                               'last_train_step':last.get('step') if last else None,'last_update':last.get('update') if last else None}
print(json.dumps(result,ensure_ascii=False))
