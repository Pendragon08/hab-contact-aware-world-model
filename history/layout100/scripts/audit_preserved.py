import argparse,hashlib,json
from pathlib import Path
root=Path('/data/code/ouyanghao/oyh/maniskill-hab-demo/results/pick_val_layout100_v1')
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
record=root/'preserved_21_before_resume.json'
if not args.verify:
    assert not record.exists()
    files=[]
    for i in range(21):
        folder=root/f'episode_{i:03d}'
        assert (folder/'episode.json').exists()
        for path in sorted(folder.iterdir()):
            if path.is_file():files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    record.write_text(json.dumps({'episode_ids':list(range(21)),'files':files},indent=2))
    print('PRESERVATION_SNAPSHOT',21,len(files))
else:
    snapshot=json.loads(record.read_text())
    for row in snapshot['files']:
        path=root/row['path']
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],path
    (root/'preserved_21_verified.json').write_text(json.dumps({'unchanged_episodes':21,'unchanged_files':len(snapshot['files']),'status':'PASS'},indent=2))
    print('PRESERVED_ALL_21_VERIFIED',len(snapshot['files']))
