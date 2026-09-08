"""Create a local, offline video gallery from measured rollout results."""
import html,json,hashlib,datetime
from html.parser import HTMLParser
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folder=root/'results/pick_val_seed0_4_v2'
summary=json.loads((folder/'summary.json').read_text())
rows=json.loads((folder/'episodes.json').read_text())
for row in rows:
    stats=json.loads((folder/f"terminal_stats_seed_{row['seed']:03d}.json").read_text())
    assert row['success_at_end']==stats['s_e']
    assert row['success_once']==stats['s_o']
    assert row['steps']==stats['l']==200
assert summary['success_at_end_count']==sum(row['success_at_end'] for row in rows)
cards=[]
for row in rows:
    seed=row['seed']; scene=json.loads((folder/f'scene_seed_{seed:03d}.json').read_text())
    status='结束时成功' if row['success_at_end'] else '结束时未成功'
    color='ok' if row['success_at_end'] else 'fail'
    video='results/pick_val_seed0_4_v2/'+row['video']
    cards.append(f'''<article><div class="row"><h2>回合 {seed+1} · 种子 {seed}</h2><span class="{color}">{status}</span></div>
<video controls preload="metadata" src="{html.escape(video)}"></video>
<p>目标：{html.escape(scene['target_object'].split('env-0_')[-1])}</p>
<p>曾经成功：{'是' if row['success_once'] else '否'} · 首次成功：{row['first_success_step'] if row['first_success_step'] is not None else '—'} 步 · 总步数：{row['steps']}</p>
<a href="{html.escape(video)}" download>保存这一回合视频</a></article>''')
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fetch 室内抓取 · ManiSkill-HAB Demo</title>
<style>body{{margin:0;background:#f4f5f7;color:#17212b;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1140px;margin:48px auto;padding:0 24px}}h1{{font-size:38px;margin:8px 0}}h2{{font-size:18px;margin:0}}.tag{{color:#637084;font-size:13px;letter-spacing:1px}}.lead{{max-width:840px;color:#48566a}}.stats{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}.stat{{background:#172e42;color:white;border-radius:14px;padding:16px 24px}}.stat b{{font-size:28px;display:block}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}}article{{background:white;padding:20px;border-radius:18px;box-shadow:0 2px 16px #15293809}}video{{display:block;width:100%;border-radius:10px;background:#192631;margin:18px 0}}.row{{display:flex;justify-content:space-between;align-items:center;gap:8px}}.ok,.fail{{font-size:13px;border-radius:20px;padding:4px 10px}}.ok{{background:#e1f5e9;color:#1b7444}}.fail{{background:#fff0d7;color:#95621c}}a{{color:#216493}}footer{{margin-top:30px;color:#637084;font-size:14px}}</style>
<main><div class="tag">MANISKILL-HAB · TIDYHOUSE / PICK · 官方预训练 SAC</div><h1>Fetch 室内抓取演示</h1>
<p class="lead">Fetch 使用移动底盘和机械臂抓取物体，并收回机械臂、保持稳定。以下为固定种子运行的全部回合。只运行抓取子任务，没有训练新模型；这些演示样本不代表完整 benchmark 成绩。</p>
<div class="stats"><div class="stat">实际运行<b>{len(rows)} 回合</b></div><div class="stat">曾经满足成功条件<b>{summary['success_once_count']} / {len(rows)}</b></div><div class="stat">结束时仍满足成功条件<b>{summary['success_at_end_count']} / {len(rows)}</b></div></div>
<div class="grid">{''.join(cards)}</div><footer>成功按原环境条件记录，包括抓住目标、收回姿态、稳定性与接触力约束。<br>
<a href="README.md">复现说明与官方资源</a> · <a href="results/pick_val_seed0_4_v2/episodes.csv">逐回合数据 CSV</a> · <a href="results/pick_val_seed0_4_v2/summary.json">原始汇总 JSON</a><br>
仿真与策略来源：<a href="https://arth-shukla.github.io/mshab/">ManiSkill-HAB 作者项目</a>。本页可离线播放本目录中的视频。</footer></main></html>'''
(root/'演示汇总.html').write_text(page,encoding='utf-8')
class LinkCheck(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value and not value.startswith(('https:','http:','#')):
                assert (root/value).is_file(),value
LinkCheck().feed(page)
files=[root/'README.md',root/'演示汇总.html']+sorted(folder.iterdir())
manifest={'created':datetime.datetime.now().astimezone().isoformat(),
          'task':'ManiSkill-HAB pretrained SAC Pick demo',
          'final_results':'results/pick_val_seed0_4_v2',
          'episodes':len(rows),'success_once_count':summary['success_once_count'],
          'success_at_end_count':summary['success_at_end_count'],
          'files':[{'path':str(p.relative_to(root)), 'bytes':p.stat().st_size,
                    'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files if p.is_file()]}
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('ARTIFACT_CHECK_PASSED: terminal statistics, all gallery links, manifest hashes')
print(root/'演示汇总.html')
