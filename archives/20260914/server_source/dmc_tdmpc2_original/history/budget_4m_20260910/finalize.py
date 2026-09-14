"""Archive completed seeds and report all available results without cherry-picking."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main(seed):
    run = ROOT / 'artifacts' / f'walker_walk_seed{seed}_4m'
    complete = json.loads((run / 'complete.json').read_text())
    assert complete['completed'] and complete['train_environment_steps'] == 4000000
    release = ROOT / 'release'
    release.mkdir(exist_ok=True)
    files = sorted(p for p in run.rglob('*') if p.is_file())
    size = sum(p.stat().st_size for p in files)
    assert shutil.disk_usage(ROOT).free >= 15*2**30 + size, 'Reserve insufficient for archive'
    bundle = release / f'{run.name}.tar.gz'
    with tarfile.open(bundle, 'x:gz', compresslevel=3) as tar:
        tar.add(run, arcname=run.name)
    save(release / f'{run.name}.manifest.json', {'files': [{'path': str(p.relative_to(run)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files], 'archive_bytes': bundle.stat().st_size, 'archive_sha256': sha(bundle)})
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    results = []
    for current in [1, 2, 3]:
        marker = ROOT / 'artifacts' / f'walker_walk_seed{current}_4m' / 'complete.json'
        if marker.exists():
            value = json.loads(marker.read_text())
            assert value['completed'] and value['train_environment_steps'] == 4000000
            results.append({'seed': current, 'return': value['final_eval']['episode_reward'], 'seconds': value['seconds']})
    reference = list(csv.DictReader((ROOT / 'reference/walker-walk.csv').open()))
    final_reference = {int(x['seed']): float(x['reward']) for x in reference if int(x['step']) == 4000000}
    summary = {'task': 'walker-walk', 'completed_seeds': results, 'paper_final_returns': final_reference,
               'all_three_complete': len(results) == 3, 'metric': 'raw return /1000; not success rate'}
    lines = ['# Walker Walk 当前结果', '', '| 种子 | 本轮最终回报 | 作者公开最终回报 |', '|---|---:|---:|']
    for item in results:
        lines.append(f"| {item['seed']} | {item['return']:.3f} | {final_reference[item['seed']]:.1f} |")
    if len(results) == 3:
        values = [x['return'] for x in results]
        mean = statistics.mean(values)
        ci = 4.3026527299 * statistics.stdev(values) / math.sqrt(3)
        summary.update(mean_return=mean, seed_mean_t95_halfwidth=ci, paper_mean_return=statistics.mean(final_reference.values()))
        lines += ['', f'三种子均值 {mean:.3f}；以种子均值为样本的t分布95%置信区间 ±{ci:.3f}（n=3，区间仅作描述）。']
    else:
        lines += ['', '仍有种子未完成，尚无完整三种子结论。']
    lines += ['', '仿真版本与官方固定配置相同，Python/PyTorch/TorchRL运行时存在已披露差异。服务器归档已生成；这不等于已经发布或完整回传。']
    save(reports / 'summary.json', summary)
    (reports / '当前结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=[1, 2, 3])
    main(parser.parse_args().seed)
