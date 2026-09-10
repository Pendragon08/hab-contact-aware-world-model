"""Render the BMPC reading report and embed only its verified local paper figures."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / 'reports/BMPC论文汇报.md'
    output = ROOT / 'BMPC论文汇报.html'
    subprocess.run([sys.executable, str(args.renderer), str(source), '--offline',
                    '--template', 'academic', '--out', str(output),
                    '--title', 'BMPC 论文汇报｜让策略与规划相互改进',
                    '--subtitle', '研究背景 · 方法原图 · 42 项任务 · 训练评估 · 结果与复现核对',
                    '--eyebrow', '论文解读 · ICLR 2025 · 中文汇报版'], check=True)
    assets = ROOT / 'reports/bmpc_presentation'
    manifest = json.loads((assets / 'evidence.json').read_text(encoding='utf-8'))
    html = output.read_text(encoding='utf-8')
    for name in ['figure1.png', 'algorithm1.png', 'figure4.png', 'figure7.png', 'tasks.png']:
        data = (assets / name).read_bytes()
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'Expected PNG: {name}')
        if hashlib.sha256(data).hexdigest() != manifest['figures'][name]['sha256']:
            raise ValueError(f'Figure hash mismatch: {name}')
        needle = f'src="bmpc_presentation/{name}"'
        if html.count(needle) != 1:
            raise ValueError(f'Expected one reference to {name}')
        html = html.replace(needle, 'src="data:image/png;base64,' + base64.b64encode(data).decode('ascii') + '"')
    css = (assets / 'presentation.css').read_text(encoding='utf-8')
    html = html.replace('</head>', '<style id="bmpc-report-layout">\n' + css + '\n</style>\n</head>', 1)
    # No JavaScript/CDN is required for reading, tables, figures or navigation.
    html = '\n'.join(line.rstrip() for line in html.splitlines()) + '\n'
    output.write_text(html, encoding='utf-8', newline='\n')
    print('BMPC report built with five verified, embedded paper figures.')


if __name__ == '__main__':
    main()
