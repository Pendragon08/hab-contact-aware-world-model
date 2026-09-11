"""Render the canonical research-topic report; no training or remote writes."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STEM='世界模型与机器人操作_高水平文献与可执行课题报告'
ASSETS=ROOT/'reports/world_model_topics_20260911'

def render(renderer,standalone_dir=None):
    source=ROOT/'reports'/(STEM+'.md')
    output=ROOT/(STEM+'.html')
    subprocess.run([sys.executable,str(renderer),str(source),'--out',str(output),'--offline',
                    '--title','世界模型与机器人操作｜高水平文献与可执行课题',
                    '--subtitle','26篇正式论文 · 5个候选方向 · 具体任务与数据 · Zotero阅读卡',
                    '--eyebrow','开题准备 · 2026年9月11日'],check=True)
    document=output.read_text(encoding='utf-8')
    document=re.sub(r'((?:href|src)=")\.\./',r'\1',document)
    css=(ASSETS/'layout.css').read_text(encoding='utf-8')
    document=document.replace('</head>','<style id="topic-report-layout">\n'+css+'\n</style>\n</head>',1)
    document=re.sub(r'(<table>.*?</table>)',r'<div class="table-scroll" tabindex="0">\1</div>',document,flags=re.S)
    document='\n'.join(line.rstrip() for line in document.splitlines())+'\n'
    output.write_text(document,encoding='utf-8',newline='\n')
    if standalone_dir:
        destination=Path(standalone_dir).resolve()
        if destination!=ROOT.parent.resolve():raise ValueError('Standalone report must sit in the project parent.')
        def link(match):
            attribute,target=match.groups()
            if target.startswith(('#','/','//')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',target):return match.group(0)
            return f'{attribute}="{ROOT.name}/{target}"'
        (destination/output.name).write_text(re.sub(r'(href|src)="([^"]+)"',link,document),encoding='utf-8',newline='\n')
    print(json.dumps({'output':str(output),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--renderer',required=True,type=Path);parser.add_argument('--standalone-dir',type=Path)
    args=parser.parse_args();render(args.renderer,args.standalone_dir)
