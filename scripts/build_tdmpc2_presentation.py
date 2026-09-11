"""Render the canonical report and embed its allowlisted local paper figure."""
import argparse
import base64
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT/'reports/TD-MPC2算法流程汇报.md'
    output = ROOT/'TD-MPC2算法流程汇报.html'
    subprocess.run([sys.executable, str(args.renderer), str(source), '--offline',
                    '--template', 'academic', '--out', str(output),
                    '--title', 'TD-MPC2 算法流程｜背景、方法与实验',
                    '--subtitle', '研究问题 · 创新贡献 · 论文原图 · 仿真平台 · 训练与评估',
                    '--eyebrow', '论文解读 · ICLR 2024 · 中文汇报版'], check=True)
    # Embed only a fixed, verified PNG after Markdown sanitization. No arbitrary URLs.
    relative = 'reports/tdmpc2_presentation/figure3.png'
    figure = (ROOT/relative).read_bytes()
    if not figure.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('Expected a PNG figure')
    html = output.read_text(encoding='utf-8')
    needle = f'src="{relative}"'
    if html.count(needle) != 1:
        raise ValueError('Expected exactly one original Figure 3')
    html = html.replace(needle, 'src="data:image/png;base64,' + base64.b64encode(figure).decode('ascii') + '"')
    css = (ROOT/'reports/tdmpc2_presentation/talk-layout.css').read_text(encoding='utf-8')
    html = html.replace('</head>', '<style id="tdmpc2-talk-layout">\n' + css + '\n</style>\n</head>', 1)
    html = '\n'.join(line.rstrip() for line in html.splitlines()) + '\n'
    output.write_text(html, encoding='utf-8', newline='\n')
    print('Embedded original Figure 3; HTML is readable offline as a single file.')

if __name__ == '__main__':
    main()
