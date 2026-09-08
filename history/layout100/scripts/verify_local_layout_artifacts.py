import argparse,hashlib,json
from pathlib import Path
from html.parser import HTMLParser
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');args=p.parse_args()
folder=root/'results/pick_val_layout100_v1'
snapshot=json.loads((folder/'preserved_21_before_resume.json').read_text())
for row in snapshot['files']:
    path=folder/row['path']
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
print('LOCAL_FIRST_21_UNCHANGED',len(snapshot['files']))
if args.final:
    manifest=json.loads((root/'layout100_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=root/row['path']
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    episodes=json.loads((folder/'episodes.json').read_text(encoding='utf-8'))
    assert len(episodes)==100 and {r['episode_id'] for r in episodes}==set(range(100))
    assert summary['success_once_count']==sum(r['success_once'] for r in episodes)
    assert summary['success_at_end_count']==sum(r['success_at_end'] for r in episodes)
    for row in episodes: assert (folder/row['video']).is_file()
    class CheckLinks(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and value and not value.startswith(('https:','http:','#')):
                    assert (root/value).is_file(),value
    CheckLinks().feed((root/'跨布局评估汇总.html').read_text(encoding='utf-8'))
    print('LOCAL_FINAL_ARTIFACTS_PASS',len(manifest['files']),'files; 100 episodes; gallery links; all hashes')
