"""Download pinned third-party inputs with archive checksums and safe extraction."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download,snapshot_download

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def extract(archive,target):
    target.mkdir(parents=True,exist_ok=True);base=target.resolve()
    with zipfile.ZipFile(archive) as z:
        members=[i for i in z.infolist() if not i.is_dir()]
        heads={i.filename.split('/')[0] for i in members}
        prefix=next(iter(heads))+'/' if len(heads)==1 and all('/' in i.filename for i in members) else ''
        for member in members:
            p=(base/member.filename[len(prefix):]).resolve()
            if not p.is_relative_to(base):raise ValueError('Unsafe archive member')
            p.parent.mkdir(parents=True,exist_ok=True)
            with z.open(member) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dependency-root',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];data=a.dependency_root/'assets/data'
    specs=[('replicacad',data/'scene_datasets/replica_cad_dataset',None),
           ('rearrange',data/'scene_datasets/replica_cad_dataset/rearrange',None),
           ('ycb',data/'assets/mani_skill2_ycb',None),
           ('pick_policy',data/'mshab_checkpoints',['rl/tidy_house/pick/all/*'])]
    for name,target,patterns in specs:
        meta=json.loads((root/f'docs/upstream/{name}_download.json').read_text())
        if meta['filename']:
            archive=Path(hf_hub_download(meta['repo'],meta['filename'],repo_type=meta['repo_type'],revision=meta['revision'],token=False))
            assert sha(archive)==meta['sha256'],name
            extract(archive,target)
        else:
            snapshot_download(meta['repo'],repo_type=meta['repo_type'],revision=meta['revision'],local_dir=target,allow_patterns=patterns,token=False)
        print('ASSET_READY',name,meta['revision'],flush=True)
    checkpoint=data/'mshab_checkpoints/rl/tidy_house/pick/all/policy.pt'
    assert sha(checkpoint)=='81f1f29b3af1ed4a3d3b502b515595dae852c4d842a1b1599203e46dee5f73e9'

if __name__=='__main__':main()
