"""Fetch completed bundles from the experiment host and verify local restoration."""
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=['contact-v1','state-pilots-v1','witnesses-v1','run-records-v1']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--remote-root',required=True);a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_@.:-]+',a.host) and re.fullmatch(r'/[A-Za-z0-9_./-]+',a.remote_root)
    assert '..' not in a.remote_root.split('/')
    subprocess.run(['ssh','-o','BatchMode=yes',a.host,f'test -f {a.remote_root}/release/READY'],check=True)
    dest=ROOT/'release';dest.mkdir(exist_ok=True)
    def fetch(name):
        assert Path(name).name==name and re.fullmatch(r'[A-Za-z0-9_.-]+',name)
        subprocess.run(['scp',f'{a.host}:{a.remote_root}/release/{name}',str(dest/name)],check=True)
    infos=[]
    for name in NAMES:
        fetch(name+'.release.json');info=json.loads((dest/(name+'.release.json')).read_text());assert info['bundle']==name;infos.append(info)
    needed=sum(i['uncompressed_bytes']+i['archive_bytes']*(2 if len(i['parts'])>1 else 1) for i in infos)+2*2**30
    if shutil.disk_usage(ROOT).free<needed:raise RuntimeError(f'Local free disk insufficient for verified extraction: need {needed/2**30:.2f} GiB')
    for info in infos:
        fetch(info['manifest']);assert sha(dest/info['manifest'])==info['manifest_sha256']
        for part in info['parts']:
            local=dest/part['name']
            if not(local.exists() and local.stat().st_size==part['bytes'] and sha(local)==part['sha256']):fetch(part['name'])
            assert local.stat().st_size==part['bytes'] and sha(local)==part['sha256']
        subprocess.run([sys.executable,str(ROOT/'scripts/restore_artifacts.py'),'--inventory',str(dest/(info['bundle']+'.release.json')),'--root',str(ROOT)],check=True)
        index=ROOT/'reports/release-index';index.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(dest/(info['bundle']+'.release.json'),index/(info['bundle']+'.release.json'))
        shutil.copyfile(dest/info['manifest'],index/info['manifest'])
    fetch('finalization.log');shutil.copyfile(dest/'finalization.log',ROOT/'reports/finalization.log')
    result={'passed':True,'bundles':NAMES,'files':sum(i['files'] for i in infos),'all_archive_and_raw_file_hashes_verified':True}
    (ROOT/'reports/local_transfer_audit.json').write_text(json.dumps(result,indent=2))
    print('ALL_BUNDLES_RESTORED',json.dumps(result),flush=True)

if __name__=='__main__':main()
