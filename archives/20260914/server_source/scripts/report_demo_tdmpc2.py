"""Summarize completed paired development evaluations without selecting a model."""
import html
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/demo_tdmpc2_v1/run'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    final=load(BASE/'complete.json');process=load(BASE/'process.json')
    assert final['completed'] and process['completed']
    out=ROOT/'reports/demo_tdmpc2_v1';out.mkdir(parents=True,exist_ok=True)
    rows=[];pred=[];videos=[]
    for step in [0,10000,20000,30000]:
        folder=BASE/f'evaluation_{step:05d}'
        summary=load(folder/'complete.json');episodes=load(folder/'episodes.json')
        assert summary['completed'] and len(episodes)==40
        assert {(r['collection_index'],r['mode']) for r in episodes}=={(i,m) for i in range(80,100) for m in ['direct_policy','mpc']}
        for mode,count in summary['counts'].items():
            group=[r for r in episodes if r['mode']==mode]
            assert count['episodes']==20
            for k in ['ever_grasp','success_once','success_at_end','force_limit_exceeded_any']:
                assert count[k]==sum(r[k] for r in group)
            rows.append({'online_step':step,'mode':mode,**count,
                         'mean_return':float(np.mean([r['return'] for r in group])),
                         'mean_action_latency_ms':float(np.mean([r['latency_mean_ms'] for r in group]))})
        for item in episodes:
            p=ROOT/item['folder']
            assert (p/'rollout.mp4').is_file() and load(p/'initial_pair_audit.json')['passed']
            with np.load(p/'trajectory.npz') as z:
                assert z['states45'].shape==(201,45) and z['effective_actions11'].shape==(200,11)
            videos.append((step,item['mode'],item['collection_index'],(p/'rollout.mp4').relative_to(ROOT).as_posix()))
        source=load(folder/'prediction_audit.json')['rows']
        for h in [1,2,3]:
            group=[r for r in source if r['horizon']==h]
            assert len(group)==20 and all(r['windows']==197 for r in group)
            pred.append({'online_step':step,'horizon':h,'windows':3940,
                         **{k:float(np.mean([r[k] for r in group])) for k in ['model_mae','persistence_mae']},
                         **{k:float(np.sqrt(np.mean([r[k]**2 for r in group]))) for k in ['model_rmse','persistence_rmse']}})
    log=[json.loads(s) for s in (BASE/'updates.jsonl').read_text().splitlines()]
    assert len(log)==35000 and sum(r['stage']=='pretrain' for r in log)==5000 and sum(r['stage']=='online' for r in log)==30000
    assert [r['online_step'] for r in log if r['stage']=='online']==list(range(1,30001))
    assert all(all(np.isfinite(v) for v in r.values() if isinstance(v,(int,float))) for r in log)
    train=list(BASE.glob('train_*/episode.json'));assert len(train)==150
    assert all(load(p)['collection_index']<80 and load(p)['steps']==200 for p in train)
    result={'completed':True,'training':final,'process':process,'evaluations':rows,'predictions':pred,
            'audit':{'passed':True,'pretrain_updates':5000,'online_updates':30000,'online_control_steps':30000,
                     'evaluation_control_steps':32000,'evaluation_videos':160,'training_episodes':150,
                     'evaluation_pair_checks':160,'development_transitions_used_for_training':0},
            'limitations':['One training seed and one task; development starts, not independent test houses.',
                           'Demonstration/BC/finite-horizon extension, not original online TD-MPC2 reproduction.',
                           'No matched demonstration-assisted SAC comparison yet; no risk module trained.',
                           'Reward prediction windows overlap; do not interpret as independent samples.']}
    (out/'training_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 示范辅助TD-MPC2训练与开发评估','',
           '完成5000次示范预训练和30000步在线交互。以下为同一训练种子、同一任务的固定开发起点评估，全部检查点均报告。',
           '', '|在线步数|控制方式|抓取/20|曾成功/20|结束成功/20|触力超限/20|', '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['online_step']}|{r['mode']}|{r['ever_grasp']}|{r['success_once']}|{r['success_at_end']}|{r['force_limit_exceeded_any']}|")
    lines+=['','## 预测检查','','|在线步数|预测步长|模型MAE|上一奖励MAE|模型RMSE|上一奖励RMSE|','|---|---|---|---|---|---|']
    for r in pred:
        lines.append(f"|{r['online_step']}|{r['horizon']}|{r['model_mae']:.6f}|{r['persistence_mae']:.6f}|{r['model_rmse']:.6f}|{r['persistence_rmse']:.6f}|")
    lines+=['','标准化只使用训练状态；开发评估不更新模型。第200步不自举未来价值，规划也按剩余回合长度截断。原HAB奖励与成功条件保持不变。',
            '', '这是一项示范辅助扩展基线。与旧BC/纯在线TD-MPC2的数据和训练预算不同，不应把差异归因于世界模型或某一组件。尚无本轮等预算示范辅助SAC，不构成完整算法优越性比较。',
            '', '原版静止判据的局限仍在；成功和累计触力指标不能替代真实机器人安全验证。',
            '', '[完整数据摘要与审计](training_summary.json)']
    (out/'训练报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    table='<table><tr><th>在线步数</th><th>控制方式</th><th>抓取/20</th><th>曾成功/20</th><th>结束成功/20</th><th>触力超限/20</th></tr>'
    for r in rows:
        table+='<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ['online_step','mode','ever_grasp','success_once','success_at_end','force_limit_exceeded_any'])+'</tr>'
    table+='</table>'
    links=''.join(f'<li><a href="{html.escape(path,quote=True)}">{step}步 · {mode} · 开发起点{idx}</a></li>' for step,mode,idx,path in videos)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>示范辅助世界模型训练</title>
<style>body{{font:17px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#14263d;background:#f4f7fb}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:10px;border:1px solid #ccd5e0}}a{{color:#165da8}}</style>
<h1>示范辅助世界模型训练与评估</h1><p>5000次示范预训练＋30000步在线交互。单任务、单训练种子、固定开发起点；所有检查点均报告。</p>{table}
<p>直接策略与MPC使用同一检查点。原HAB成功条件保持不变，未训练风险模块。此结果不代表独立最终benchmark。</p>
<p><a href="reports/demo_tdmpc2_v1/训练报告.md">详细训练报告</a></p><details><summary>全部160段评估视频</summary><ul>{links}</ul></details></html>'''
    (ROOT/'示范辅助世界模型训练汇总.html').write_text(page,encoding='utf-8')
    print('DEMO_TDMPC2_REPORT_PASS',json.dumps(result['audit']),flush=True)


if __name__=='__main__':main()
