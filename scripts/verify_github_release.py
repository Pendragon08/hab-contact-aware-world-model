"""Check GitHub's uploaded size and SHA256 for every expected Release asset."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--bundles',nargs='+',required=True)
    p.add_argument('--extra',nargs='*',default=[]);a=p.parse_args()
    response=json.loads(subprocess.check_output(['gh','api',f'repos/Pendragon08/hab-contact-aware-world-model/releases/tags/{a.tag}'],text=True,encoding='utf-8'))
    assets={x['name']:x for x in response['assets']};expected=[]
    for name in a.bundles:
        inventory=ROOT/'release'/f'{name}.release.json';info=json.loads(inventory.read_text())
        expected+=[inventory,ROOT/'release'/info['manifest']]+[ROOT/'release'/p['name'] for p in info['parts']]
    expected += [ROOT/e for e in a.extra]
    rows=[]
    for path in expected:
        actual=assets[path.name];digest=sha(path)
        assert actual['state']=='uploaded' and actual['size']==path.stat().st_size,path.name
        assert actual['digest']=='sha256:'+digest,path.name
        rows.append({'name':path.name,'size':actual['size'],'sha256':digest,'github_digest':actual['digest']})
    result={'passed':True,'tag':a.tag,'release_url':response['html_url'],'is_draft':response['draft'],'assets_verified':rows}
    (ROOT/'reports'/f'github_{a.tag}_audit.json').write_text(json.dumps(result,indent=2))
    print('GITHUB_RELEASE_VERIFIED',len(rows),response['html_url'])

if __name__=='__main__':main()
