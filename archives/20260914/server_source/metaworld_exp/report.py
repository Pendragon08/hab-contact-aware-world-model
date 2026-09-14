"""Read recorded evidence only; generate compact JSON, Markdown and HTML."""
import html,json,os
from pathlib import Path
ROOT=Path(os.environ.get('MW_ROOT',Path(__file__).resolve().parent))
BASE=ROOT/'artifacts/pick_place_v1'
out=ROOT/'reports';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
collection=read(BASE/'collection/complete.json');witness=read(BASE/'witness/complete.json')
run=read(BASE/'run/complete.json');process=read(BASE/'run/process.json')
rows=[]
for f in sorted((BASE/'run').glob('evaluation_*/complete.json'))+list((BASE/'run').glob('final_test/complete.json')):
    d=read(f)
    for mode,c in d['counts'].items():rows.append({'split':d['split'],'step':d['online_step'],'mode':mode,**c})
status={'collection':collection,'witness_passed':bool(witness and witness['passed']),'training_complete':bool(run and run['completed']),'process':process,'evaluations':rows}
(out/'summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=['# Meta-World 抓取放置实验报告','',
      '实验：Meta-World 3.1.1 pick-place-v3，状态输入，示范辅助 TD-MPC2，GPU1。此为单任务单训练种子先导实验，并非原论文复现或完整 MT1/MT50。','']
if collection:text += [f"脚本专家示范：{collection['episodes']} 条、{collection['control_steps']:,} 步；曾成功 {collection['success_once']}/{collection['episodes']}，末步成功 {collection['success_at_end']}/{collection['episodes']}。这些是专家的数据质量指标，不是学习模型的成绩。",'']
text += [f"工程验证：{'通过' if status['witness_passed'] else '尚未通过'}。正式训练：{'完成' if status['training_complete'] else '尚未确认完成'}。",'']
if process and not process['completed']:text += ['运行中断：`'+process['error']+'`。保留记录，不自动重跑。','']
text += ['| 集合 | 在线步数 | 控制方式 | 曾成功 | 末步成功 | 曾抓住 | 平均回报 |','|---|---:|---|---:|---:|---:|---:|']
for r in rows:
    n=r['episodes'];text.append(f"| {r['split']} | {r['step']} | {r['mode']} | {r['success_once']}/{n} | {r['success_at_end']}/{n} | {r['grasp_once']}/{n} | {r['return_mean']:.2f} |")
text += ['','每次轨迹500步，保留环境原生奖励与成功判定。末步成功与曾成功分开报告。开发集不能当作最终测试；最终测试只使用固定100,000步权重。','',
         '预算：40,000步专家采集、10,000步学生纠正采集、5,000次预训练更新、100,000步在线训练。开发评估80,000步、最终评估20,000步单列。','',
         '当前对照只能检验规划是否帮助同一个已训练模型；还不能证明优于原版TD-MPC2。后续需要匹配数据/算力预算的基线和多训练种子。','',
         '## 视频','']
videos=sorted(BASE.glob('run/evaluation_*/*/rollout.mp4'))+sorted(BASE.glob('run/final_test/*/rollout.mp4'))
for p in videos:text.append(f'- [{p.parent.parent.name}/{p.parent.name}]({os.path.relpath(p,out).replace(os.sep,"/")})')
(out/'实验报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
cards=''.join(f'<h3>{html.escape(p.parent.parent.name+" / "+p.parent.name)}</h3><video controls preload="metadata" width="480" src="{html.escape(os.path.relpath(p,out).replace(os.sep,"/"))}"></video>' for p in videos)
doc='<!doctype html><meta charset="utf-8"><title>Meta-World 实验汇总</title><style>body{max-width:1100px;margin:30px auto;font:17px/1.7 system-ui;background:#f4f7fa;color:#152334}pre{white-space:pre-wrap;background:white;padding:24px;border-radius:14px}video{max-width:100%}</style><pre>'+html.escape('\n'.join(text))+'</pre>'+cards
(out/'演示汇总.html').write_text(doc,encoding='utf-8')
print(json.dumps(status,ensure_ascii=False))
