"""Render the canonical proposal meeting Markdown; no training or network access."""
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = '开题方向讨论汇报_世界模型动作后果排序'
ASSETS = ROOT / 'reports/proposal_meeting_20260911'

def render(renderer, standalone_dir=None):
    source = ROOT / 'reports' / (STEM + '.md')
    output = ROOT / (STEM + '.html')
    subprocess.run([sys.executable, str(renderer), str(source), '--out', str(output),
                    '--offline', '--title', '开题方向讨论｜世界模型动作后果排序',
                    '--subtitle', '研究什么 · 已有基础 · 分阶段方案 · 请老师判断可行性',
                    '--eyebrow', '开题讨论 · 2026年9月11日'], check=True)
    document = output.read_text(encoding='utf-8')
    # Source links resolve from reports/; HTML sits at the repository root.
    document = re.sub(r'((?:href|src)=")\.\./', r'\1', document)
    css = (ASSETS / 'layout.css').read_text(encoding='utf-8')
    document = document.replace('</head>', '<style id="proposal-meeting-layout">\n' + css + '\n</style>\n</head>', 1)
    document = '\n'.join(line.rstrip() for line in document.splitlines()) + '\n'
    output.write_text(document, encoding='utf-8', newline='\n')
    if standalone_dir:
        destination = Path(standalone_dir).resolve()
        if destination != ROOT.parent.resolve():
            raise ValueError('Standalone copy is supported only in the project parent directory.')
        def relative_link(match):
            attribute, target = match.groups()
            if target.startswith(('#', '/', '//')) or re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                return match.group(0)
            return f'{attribute}="{ROOT.name}/{target}"'
        copy = re.sub(r'(href|src)="([^"]+)"', relative_link, document)
        (destination / output.name).write_text(copy, encoding='utf-8', newline='\n')
    print(json.dumps({'output':str(output),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', type=Path, required=True)
    parser.add_argument('--standalone-dir', type=Path)
    args = parser.parse_args()
    render(args.renderer, args.standalone_dir)
