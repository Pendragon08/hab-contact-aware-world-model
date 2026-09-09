"""Read-only compact status; never launches, resumes, or changes an experiment."""
import argparse
import json
from pathlib import Path


def status(root):
    base=root/'artifacts/demo_tdmpc2_v1'
    result={'training_started':(base/'run/provenance.json').exists(),'evaluations':{}}
    for stage in ['witness','run']:
        for name in ['process','complete']:
            p=base/stage/(name+'.json')
            if p.exists():result[stage+'_'+name]=json.loads(p.read_text())
    p=base/'run/updates.jsonl'
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-32768));lines=f.read().splitlines()
        for line in reversed(lines):
            try:result['last_update']=json.loads(line);break
            except (ValueError,UnicodeDecodeError):continue
    for folder in sorted((base/'run').glob('evaluation_*')):
        p=folder/'complete.json'
        if p.exists():result['evaluations'][folder.name]=json.loads(p.read_text())
        elif (folder/'episodes.json').exists():
            rows=json.loads((folder/'episodes.json').read_text())
            result['evaluations'][folder.name]={'completed':False,'completed_episodes':len(rows),'last_episode':rows[-1]}
    result['completed_online_episodes']=len(list((base/'run').glob('train_*/episode.json')))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print(json.dumps(status(a.root),ensure_ascii=False,indent=2))
