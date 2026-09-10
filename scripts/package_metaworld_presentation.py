"""Package only the presentation and its explicit local links/media."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib, json, zipfile, argparse
ROOT=Path(__file__).resolve().parents[1]
NAME='MetaWorld_TD-MPC2复现与实验结果汇报.html'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.links.append(v)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    doc=ROOT/NAME
    links=Links();links.feed(doc.read_text(encoding='utf-8'))
    paths={doc,ROOT/'reports/MetaWorld_TD-MPC2复现与实验结果汇报.md',ROOT/'reports/TD-MPC2算法流程汇报.md'}
    for url in links.links:
        parsed=urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:continue
        target=(ROOT/unquote(parsed.path)).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():raise ValueError(f'Missing or out of scope: {url}')
        paths.add(target)
    review=ROOT/(NAME+'.review.json')
    if review.exists():paths.add(review)
    manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('使用说明.txt','请先完整解压，然后双击根目录“'+NAME+'”。保持目录结构，视频可离线播放。正文内论文/GitHub链接需要联网。PDF仅保留静态内容。包内结果为2026-09-09完成的单任务单种子缩减预算复现，不是新算法结果。附带的代码只供查阅，不是完整训练环境；不含训练权重、全部轨迹或第三方仿真资产。')
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for rel,v in manifest.items():assert hashlib.sha256(z.read(rel)).hexdigest()==v['sha256']
    print(json.dumps({'files':len(paths),'zip_bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
