"""Audit all paired attempts and generate development-only tables and media."""
import csv, hashlib, html, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(os.environ['HAB_ROOT']); OUT=ROOT/'artifacts/contact_v1'; REPORT=ROOT/'reports/contact_v1'
METHODS=['sac','conservative','force_rule']
LABELS={'sac':'Original SAC','conservative':'Conservative','force_rule':'Force rule'}

def save(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def paired_interval(baseline,other,seed=91008):
    rng=np.random.default_rng(seed); d=np.asarray(other,dtype=float)-np.asarray(baseline,dtype=float)
    # Stratify by the five fixed layouts; no claim of between-house uncertainty.
    samples=np.empty(10000)
    for i in range(10000):
        idx=np.concatenate([rng.integers(s*20,(s+1)*20,size=20) for s in range(5)])
        samples[i]=d[idx].mean()
    return {'difference':float(d.mean()),'paired_stratified_bootstrap_95':np.quantile(samples,[.025,.975]).tolist(),
            'other_only':int(np.sum(d==1)), 'baseline_only':int(np.sum(d==-1))}

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    rows=[]; events={}; initial={}; videos=[]
    plan=json.loads((ROOT/'configs/contact_v1.json').read_text())
    for method in METHODS:
        for i in range(100):
            ep=OUT/method/f'episode_{i:03d}'
            row=json.loads((ep/'episode.json').read_text()); init=json.loads((ep/'initial.json').read_text())
            step=[json.loads(x) for x in (ep/'steps.jsonl').read_text().splitlines()]
            terminal=json.loads((ep/'terminal.json').read_text())
            assert row['method']==method and row['pair_id']==i and row['steps']==len(step)==terminal['l']==200
            assert row['layout']==plan['layouts'][i//20] and row['seed']==60000+100*(i//20)+i%20
            assert row['success_once']==bool(terminal['s_o'])==any(x['success'] for x in step)
            assert row['success_at_end']==bool(terminal['s_e'])==step[-1]['success']
            for j,e in enumerate(step,1):
                assert e['step']==j and e['success']==all(e[k] for k in ['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit'])
                previous=step[j-2] if j>1 else {'robot_force':init['initial_force'],'robot_cumulative_force':init['initial_cumulative']}
                assert e['input_force']==previous['robot_force'] and e['input_cumulative_force']==previous['robot_cumulative_force']
            with np.load(ep/'trajectory.npz') as t:
                assert t['observed_states'].shape[0]==201 and t['executed_actions'].shape==(200,13)
                np.testing.assert_array_equal(t['actor_actions'][:,7],t['executed_actions'][:,7])
                assert np.all(t['executed_actions'][:,8:10]==0)
                assert all(np.isfinite(t[k]).all() for k in t.files)
                assert t['fetch_head_depth'].shape[0]==t['fetch_hand_depth'].shape[0]==201
            if i not in initial: initial[i]=init
            else: assert initial[i]==init
            reader=imageio.get_reader(ep/'rollout.mp4'); count=sum(1 for _ in reader); reader.close()
            assert count==201
            videos.append({'method':method,'pair_id':i,'frames':count})
            rows.append(row); events[method,i]=step
    grouped={m:[r for r in rows if r['method']==m] for m in METHODS}
    summary={}
    for m,rr in grouped.items():
        summary[m]={'episodes':len(rr)}
        for k in ['success_once','success_at_end','force_exceeded_before_or_at_first_success','force_limit_exceeded_any','ever_grasped']:
            summary[m][k+'_count']=sum(r[k] for r in rr)
        for k in ['first_success_step','cumulative_force_at_first_success','final_cumulative_force','actor_ms_mean','modifier_ms_mean','mean_action_scale']:
            v=[r[k] for r in rr if r[k] is not None]
            summary[m][k+'_mean']=float(np.mean(v)) if v else None
        summary[m]['reset_plus_rollout_seconds']=sum(r['reset_seconds']+r['rollout_seconds'] for r in rr)
    paired={m:{k:paired_interval([r[k] for r in grouped['sac']],[r[k] for r in grouped[m]])
                for k in ['success_once','success_at_end','force_exceeded_before_or_at_first_success']}
            for m in METHODS[1:]}
    save(REPORT/'summary.json',{'scope':'development-only; five furniture layouts in one validation macro-scene',
                              'methods':summary,'paired':paired,'formal_episodes':300,'formal_steps':60000})
    save(REPORT/'episodes.json',rows); save(REPORT/'video_audit.json',videos)
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
    with (REPORT/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,['success_once_count','success_at_end_count','force_exceeded_before_or_at_first_success_count'],
                            ['Success at least once','Success at step 200','Force limit before success']):
        values=[summary[m][key] for m in METHODS]
        ax.bar([LABELS[m] for m in METHODS],values,color=['#355c7d','#e9a23b','#27a696'])
        for i,v in enumerate(values): ax.text(i,v+1,str(v),ha='center')
        ax.set_ylim(0,110); ax.set_title(title); ax.set_ylabel('Episodes / 100'); ax.tick_params(axis='x',rotation=18)
    fig.tight_layout(); fig.savefig(REPORT/'comparison.png',dpi=160); fig.savefig(REPORT/'comparison.pdf'); plt.close(fig)
    # Deterministic, outcome-transparent selection: lowest ID per pattern; no cherry-picked claim.
    patterns={'rule_only_success':lambda i:not grouped['sac'][i]['success_once'] and grouped['force_rule'][i]['success_once'],
              'baseline_only_success':lambda i:grouped['sac'][i]['success_once'] and not grouped['force_rule'][i]['success_once'],
              'all_success':lambda i:all(grouped[m][i]['success_once'] for m in METHODS),
              'all_fail':lambda i:all(not grouped[m][i]['success_once'] for m in METHODS)}
    examples={k:next((i for i in range(100) if pred(i)),None) for k,pred in patterns.items()}
    save(REPORT/'example_selection.json',{'rule':'lowest pair ID in each pattern; all steps unretimed','examples':examples})
    selected=list(dict.fromkeys(i for i in examples.values() if i is not None))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    with imageio.get_writer(OUT/'comparison_demo.mp4',fps=20,codec='libx264',quality=7) as writer:
        for i in selected:
            readers=[imageio.get_reader(OUT/m/f'episode_{i:03d}/rollout.mp4') for m in METHODS]
            for frame_idx,frames in enumerate(zip(*readers)):
                canvas=Image.new('RGB',(1536,592),'#101827'); draw=ImageDraw.Draw(canvas)
                for col,(m,fr) in enumerate(zip(METHODS,frames)):
                    canvas.paste(Image.fromarray(fr),(col*512,80))
                    e=events[m,i][max(0,frame_idx-1)]
                    draw.text((col*512+10,5),f'{LABELS[m]} | Pair {i} | Step {frame_idx}',font=font,fill='white')
                    if frame_idx:
                        draw.text((col*512+10,30),f"Success now: {e['success']}  Scale: {e['scale']:.2f}",font=font,fill='white')
                        draw.text((col*512+10,54),f"Cumulative contact: {e['robot_cumulative_force']:.0f} / 5000",font=font,fill='white')
                    else: draw.text((col*512+10,30),'Initial state (no action executed)',font=font,fill='white')
                writer.append_data(np.asarray(canvas))
            for r in readers:r.close()
    lines=['# 接触控制配对开发实验报告','',
           '本轮比较冻结官方 SAC 与两种简单控制改动。参数在运行前固定，使用新种子；属于开发实验，不能作为最终泛化性能。',
           '', '|方法|曾成功 /100|结束成功 /100|成功前或未成功时触力超限 /100|任意时刻触力超限 /100|', '|---|---:|---:|---:|---:|']
    for m in METHODS:
        s=summary[m]; lines.append(f"|{LABELS[m]}|{s['success_once_count']}|{s['success_at_end_count']}|{s['force_exceeded_before_or_at_first_success_count']}|{s['force_limit_exceeded_any_count']}|")
    lines+=['','![三方案对比](comparison.png)','','## 配对差异与不确定性','',
            '区间为固定五个布局内分层配对 bootstrap（10,000 次）；不包含独立房屋或训练随机种子的变化。', '']
    for m in METHODS[1:]:
        p=paired[m]['success_once']; lo,hi=p['paired_stratified_bootstrap_95']
        lines.append(f"- {LABELS[m]} 相对 SAC：曾成功率变化 {100*p['difference']:+.1f} 个百分点，95% 区间 [{100*lo:+.1f}, {100*hi:+.1f}]；仅改动方案成功 {p['other_only']} 回合，仅 SAC 成功 {p['baseline_only']} 回合。")
    lines+=['','## 方法与完整性','',
            '- 每方法 100 回合 × 200 步，共 60,000 正式控制步；另存独立配对验证，不混入统计。',
            '- 每一对均校验初始完整观测、模拟器和控制器状态完全相同；每次重新配置并指定布局。重复基线探针用于检验重置可复现性，不代表任意中途状态恢复已解决。',
            '- 夹爪绝对位置命令保持原值；机械臂增量、躯干增量和底盘速度可缩放/滤波；头部与官方设置一样固定。',
            '- 力规则仅读取上一步返回的接触量与累计量；它额外使用了模拟器力信号，相对原视觉 SAC 属于诊断性控制比较。',
            '- 成功判据与环境原始判据一致，没有提前终止或冻结物理仿真；成功后继续到第 200 步的结果单独解释。',
            '- 完成步数及成功时接触量均为成功回合条件均值，方案间成功回合集合不同，不作无条件速度优势解释。',
            '- 保存 201 个状态观测、两路最新深度帧、初始三帧堆叠、200 个原动作和实际动作、逐步指标、初始状态及完整视频。',
            '- 数据来自同一验证宏观场景的五个家具布局，物体采样不平衡；不是完整 HAB benchmark。',
            '- 失败分类是观测到的终局条件，不是已证实的因果解释。本实验没有训练世界模型。',
            '', '## 对比演示','',
            '演示按最低回合编号选择规则独有成功、基线独有成功、共同成功、共同失败模式（不存在则跳过），三方案并排、播放速度相同。完整视频均已解码核验。',
            '', '视频及原始数据见同仓库 Release 的 contact_v1 归档。']
    (REPORT/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cards=[]
    for i in range(100):
        cells=[]
        for m in METHODS:
            r=grouped[m][i]
            cells.append(f'<div><b>{LABELS[m]}</b><p>曾成功 {r["success_once"]} · 结束成功 {r["success_at_end"]}</p><video controls preload="none" src="artifacts/contact_v1/{m}/episode_{i:03d}/rollout.mp4"></video></div>')
        cards.append(f'<section><h2>配对 {i} · 布局 {grouped["sac"][i]["layout"]} · 种子 {grouped["sac"][i]["seed"]}</h2><div class="row">'+''.join(cells)+'</div></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HAB 接触控制配对实验</title><style>body{font-family:system-ui;background:#101827;color:#e7edf5;margin:30px}h1{color:#64d9cb}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}section{background:#1d293b;margin:20px 0;padding:18px;border-radius:12px}img{max-width:100%}a{color:#64d9cb}</style><h1>HAB 接触控制配对开发实验</h1><p>冻结官方 SAC / 保守运动 / 接触力规则。100 个相同初始条件，300 回合。开发结果，未训练世界模型。</p><p>下方完整回合均保存，视频可暂停、拖动或全屏。主指标为曾经成功率，结束时成功为额外诊断。</p><img src="reports/contact_v1/comparison.png"><h2>同步对比演示</h2><video controls preload="metadata" src="artifacts/contact_v1/comparison_demo.mp4"></video>'''+''.join(cards)+'</html>'
    (ROOT/'接触控制对比汇总.html').write_text(page,encoding='utf-8')
    save(REPORT/'audit.json',{'passed':True,'paired_initial_conditions':100,'episodes':300,'steps':60000,'videos_decoded':300,'frames_each':201})
    print('CONTACT_REPORT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
