"""Audit and present bounded frozen-policy diagnostics without changing prior results."""
import hashlib,html,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/bc_diagnosis_v1';OUT=ROOT/'reports/bc_diagnosis_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False)
plan=load(ROOT/'configs/bc_diagnosis_v1.json')
assert sha(ROOT/plan['checkpoint'])==plan['checkpoint_sha256']
for stage in ['witness','fit']:
    assert load(BASE/f'process_{stage}.json')['completed']
    record=load(BASE/f'provenance_{stage}.json')
    assert record['source_sha256']==sha(BASE/'source_snapshot/scripts/bc_diagnosis.py')
    assert record['config_sha256']==sha(ROOT/'configs/bc_diagnosis_v1.json')
    assert record['checkpoint_sha256']==plan['checkpoint_sha256']
assert not load(BASE/'process_offline.json')['completed']
for stage in ['serial_witness','offline_serial']:
    assert load(BASE/f'process_{stage}.json')['completed']
    record=load(BASE/f'provenance_{stage}.json')
    assert record['source_sha256']==sha(BASE/'serial_source/bc_diagnosis_serial.py')
    assert record['source_parent_sha256']==sha(BASE/'source_snapshot/scripts/bc_diagnosis.py')
    assert record['config_sha256']==sha(ROOT/'configs/bc_diagnosis_v1.json')
    assert record['checkpoint_sha256']==plan['checkpoint_sha256']
assert load(BASE/'process_witness.json')['wall_seconds']+load(BASE/'process_serial_witness.json')['wall_seconds']<=300
assert load(BASE/'process_offline.json')['wall_seconds']+load(BASE/'process_offline_serial.json')['wall_seconds']<=900
assert load(BASE/'process_fit.json')['wall_seconds']<=1200
precision=load(BASE/'serial_witness/complete.json')
assert precision['passed'] and precision['threshold_unchanged']==plan['teacher_replay_tolerance']
for row in precision['rows']:
    raw=np.load(BASE/f"serial_witness/{row['source']}.npz")
    assert raw['reference'].shape==raw['batch1'].shape==raw['batch8'].shape==(200,13)
    assert all(np.isfinite(raw[k]).all() for k in raw.files)
    assert float(np.abs(raw['batch1']-raw['reference']).max())==row['serial_max_error']
    assert float(np.abs(raw['batch8']-raw['reference']).max())==row['batch8_max_error']
fit=load(BASE/'fit/episodes.json');old=ROOT/'artifacts/bc_v1_2'
dev=load(old/'student/episodes.json');teachers=load(old/'collection/episodes.json')
assert len(fit)==len(dev)==20
assert [r['collection_index'] for r in fit]==plan['fit_collection_indices']
assert len({r['spawn_idx'] for r in fit})==20
assert {r['spawn_idx'] for r in fit}.isdisjoint({r['spawn_idx'] for r in dev})
checks=['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit']
video_counts=[]
for r in fit:
    folder=ROOT/r['folder'];data=np.load(folder/'trajectory.npz')
    assert sha(folder/'trajectory.npz')==r['trajectory_sha256']
    assert all(np.isfinite(data[k]).all() for k in data.files)
    assert data['states45'].shape==(201,45)
    steps=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==r['steps']==200
    assert all(e['success']==all(e[k] for k in checks) for e in steps)
    assert r['success_once']==any(e['success'] for e in steps)
    assert r['success_at_end']==steps[-1]['success']
    assert r['ever_grasp']==any(e['is_grasped'] for e in steps)
    assert load(folder/'initial_pair_audit.json')['passed']
    terminal=load(folder/'terminal.json');assert terminal['l']==200 and bool(terminal['s_e'])==r['success_at_end']
    reader=imageio.get_reader(folder/'rollout.mp4');n=sum(1 for _ in reader);reader.close();assert n==201
    video_counts.append({'folder':r['folder'],'frames':n})
analysis=load(BASE/'offline_serial/episodes.json');assert len(analysis)==80
grouped={k:[r for r in analysis if r['group']==k] for k in ['teacher_fit','student_fit','teacher_dev','student_dev']}
stats={};time_curves={}
for group,rows in grouped.items():
    assert len(rows)==20
    motion=[];grip=[];distance=[]
    for r in rows:
        assert sha(ROOT/r['source_folder']/'trajectory.npz')==r['source_sha256']
        data=np.load(ROOT/r['analysis_file'])
        assert all(np.isfinite(data[k]).all() for k in data.files)
        motion.append(data['motion_mse']);grip.append(data['gripper_mse']);distance.append(data['nearest_fit_state_rms'])
    motion=np.stack(motion);grip=np.stack(grip);distance=np.stack(distance)
    stats[group]={k:sum(r[k] for r in rows) for k in ['ever_grasp','success_once','success_at_end']}
    stats[group].update({'episodes':20,'motion_mse_mean':float(motion.mean()),'gripper_mse_mean':float(grip.mean()),
            'total_mse_mean':float((motion+grip).mean()),'early20_total_mse':float((motion+grip)[:,:20].mean()),
            'nearest_distance_median':float(np.median(distance)),'nearest_distance_early20_median':float(np.median(distance[:,:20])),
            'force_limit_episodes':sum(r['first_force_limit_step'] is not None for r in rows)})
    time_curves[group]=np.median(distance,axis=0).tolist()
replay_max=max(r['teacher_replay_max_error'] for r in analysis if r['teacher_replay_max_error'] is not None)
audit={'passed':True,'new_full_episodes':20,'new_physical_steps':4000,'offline_episodes':80,'updates':0,
       'original_development_episodes_reused_without_retry':20,'video_counts':video_counts,
       'teacher_replay_max_error':replay_max,'checkpoint_sha256':sha(ROOT/plan['checkpoint']),
       'serial_precision_witness':precision['rows'],'control_steps_note':'4000 formal env.step calls, plus 8 initial witness calls; not internal physics integrator substeps.'}
save(OUT/'audit.json',audit);save(OUT/'summary.json',{'groups':stats,'median_distance_by_step':time_curves,
      'selection_rule':plan['selection_rule'],'scope':plan['notes'],
      'stages':{stage:load(BASE/f'process_{stage}.json') for stage in ['witness','fit','offline','serial_witness','offline_serial']}})
fig,axes=plt.subplots(1,3,figsize=(15,4))
pos=np.arange(3)
for offset,key,label in [(-.18,'student_fit','BC: fitted starts'),(.18,'student_dev','BC: held-out starts')]:
    axes[0].bar(pos+offset,[stats[key][k] for k in ['ever_grasp','success_once','success_at_end']],.36,label=label)
axes[0].set_xticks(pos,['Ever grasp','Success once','End success']);axes[0].set_ylim(0,20);axes[0].legend(fontsize=8);axes[0].set_ylabel('Episodes / 20')
labels=['Teacher fit','Student fit','Teacher dev','Student dev'];keys=list(grouped)
axes[1].bar(np.arange(4),[stats[k]['total_mse_mean'] for k in keys]);axes[1].set_xticks(np.arange(4),labels,rotation=22);axes[1].set_ylabel('BC vs teacher action MSE (motion + gripper)')
for k,label in zip(keys,labels):axes[2].plot(time_curves[k],label=label)
axes[2].set(xlabel='Step before action',ylabel='Median nearest fitted-state RMS');axes[2].legend(fontsize=8)
fig.suptitle('Frozen BC diagnosis: fit starts selected from successful demonstrations')
fig.tight_layout();fig.savefig(OUT/'diagnosis.png',dpi=150);fig.savefig(OUT/'diagnosis.pdf');plt.close(fig)

chosen_fit=next((r for r in fit if r['success_once']),fit[0])
chosen_dev=next(r for r in dev if not r['success_once'])
pairs=[('FITTED START',teachers[chosen_fit['collection_index']],chosen_fit),
       ('HELD-OUT START',next(r for r in teachers if r['seed']==chosen_dev['seed']),chosen_dev)]
writer=imageio.get_writer(OUT/'diagnosis_demo.mp4',fps=20,codec='libx264',quality=7)
try:
    for label,t,s in pairs:
        left=imageio.get_reader(ROOT/t['folder']/'rollout.mp4');right=imageio.get_reader(ROOT/s['folder']/'rollout.mp4')
        for i,(a,b) in enumerate(zip(left,right)):
            canvas=Image.new('RGB',(1024,576),'#12243b');draw=ImageDraw.Draw(canvas)
            draw.text((12,8),f"{label} | seed {s['seed']} | frame {i}/200",fill='white')
            draw.text((12,34),'Official teacher',fill='white')
            draw.text((524,34),f"Frozen BC | once={s['success_once']} end={s['success_at_end']}",fill='white')
            canvas.paste(Image.fromarray(a),(0,64));canvas.paste(Image.fromarray(b),(512,64));writer.append_data(np.asarray(canvas))
        left.close();right.close()
finally:writer.close()
reader=imageio.get_reader(OUT/'diagnosis_demo.mp4');fps=reader.get_meta_data()['fps'];frames=[];count=0
for i,f in enumerate(reader):
    assert f.shape==(576,1024,3)
    if i%201==100:frames.append(Image.fromarray(f))
    count+=1
reader.close();assert count==402 and fps==20
preview=Image.new('RGB',(1024,1152))
for i,f in enumerate(frames):preview.paste(f,(0,576*i))
preview.save(OUT/'demo_preview.png')
save(OUT/'demo_audit.json',{'passed':True,'frames':count,'fps':fps,'selection':[{'label':a,'teacher':b['folder'],'student':c['folder']} for a,b,c in pairs],
                         'rule':'first success-once fitted student if any, else first fitted student; first no-success held-out student'})

sf=stats['student_fit'];sd=stats['student_dev'];tf=stats['teacher_fit'];td=stats['teacher_dev']
release='https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.3.1-bc-diagnosis'
lines=['# 固定行为克隆策略的闭环失败诊断','',
       f"本轮没有训练新模型。固定第5000次更新的 BC，新增20个已用于训练的起点测试，曾抓取 {sf['ever_grasp']}/20、曾完整成功 {sf['success_once']}/20、结束完整成功 {sf['success_at_end']}/20。原20个留出结果保留为 4/20、3/20、1/20，没有重试。",'',
       '## 实验如何区分问题','',
       '从53条结束成功的拟合示范中，按固定顺序等间距选20条；所选起点确实参与了原模型训练。每次运行完整200步，教师只给旁路建议，全部动作仍由学生执行。新的8步独立预检确认旁路记录未改变原学生轨迹。',
       '离线重建80条轨迹的教师观测：20条教师拟合、20条学生拟合、20条教师留出、20条学生留出。初始三帧深度从完整快照恢复，后续使用逐步深度记录。原教师输出和现场旁路建议用于核验重建；所有建议标签只用于诊断，特别是留出标签禁止进入训练。','',
       '## 闭环结果','',
       '|数据组|曾抓取/20|曾完整成功/20|结束完整成功/20|任意时刻累计触力超限/20|','|---|---:|---:|---:|---:|']
names={'teacher_fit':'原教师：选定拟合起点','student_fit':'学生：选定拟合起点','teacher_dev':'原教师：全部留出起点','student_dev':'学生：全部留出起点'}
for key,g in stats.items():lines.append(f"|{names[key]}|{g['ever_grasp']}|{g['success_once']}|{g['success_at_end']}|{g['force_limit_episodes']}|")
lines+=['','拟合起点按教师成功示范筛选，教师拟合组20/20结束成功是选择规则的结果。留出组未如此筛选，因此不能将两组差距直接当成无偏的泛化估计。这里检验的是：学生是否连熟悉、已模仿过的起点也难以完成。','',
        '## 动作分歧与状态偏离','',
        '|数据组|运动MSE|夹爪MSE|两者之和|前20步两者之和|最近拟合状态距离中位数|','|---|---:|---:|---:|---:|---:|']
for k,g in stats.items():lines.append(f"|{names[k]}|{g['motion_mse_mean']:.5f}|{g['gripper_mse_mean']:.5f}|{g['total_mse_mean']:.5f}|{g['early20_total_mse']:.5f}|{g['nearest_distance_median']:.5f}|")
lines+=['','![固定策略诊断](diagnosis.png)','',
        '动作分歧在相同已记录观测上比较学生与冻结教师。运动十维平均MSE与夹爪一维MSE等权相加；教师的建议不是唯一正确答案。最近邻距离使用原训练统计标准化的前42维状态，到全部10600个成功拟合状态的最小欧氏距离除以√42；不含附加力、累积力和时间。它是状态偏离代理值，不是概率、风险评分或因果证据。','',
        '## 解释与下一步','',
        ('学生在已参与训练的起点上仍明显低于教师，说明困难不只发生在新的初始位置：低开环模仿误差没有保证闭环纠错。' if sf['success_once']<15 else '学生在拟合起点的完成情况明显改善，但仍需考虑按教师成功筛选起点造成的偏差，不能单凭这一差距证明泛化机制。'),
        f"原教师拟合轨迹上的动作MSE为 {tf['total_mse_mean']:.5f}，学生实际访问的拟合轨迹上为 {sf['total_mse_mean']:.5f}。这是同一冻结模型在不同访问状态上的描述性比较；两组状态由不同动作生成，并非受控因果实验。",
        '当前45维输入缺少显式底盘速度和深度历史；教师使用三帧头部/手部深度。观测信息不足、动作多解和闭环分布偏移尚未被分别隔离。本轮的距离和动作分歧只能为后续实验提供依据，不能宣称已找到唯一失败原因。',
        '下一轮优先做受控纠错实验：只在训练起点采集学生访问状态上的教师标签，保持模型输入、优化预算与评估协议一致，对比加入这些纠错样本与等量原教师轨迹样本。现有留出标签不进入训练。若纠错仍不足，再单独比较短历史/底盘速度或深度输入；不要同时更换多个因素。本轮没有执行这些训练，也未启动世界模型/风险规划。','',
        '## 可复核性与交付','',f"原教师离线动作重建最大误差 {replay_max:.3g}；20个新回合共4000物理步、20段201帧视频以及80条离线诊断记录均核验。模型 SHA256 始终为 `{plan['checkpoint_sha256']}`。原 HAB 主指标与静止辅助诊断的区别继续沿用上一版说明，没有修改判据。",'',
        '- [固定方案与执行入口](../../docs/BC_DIAGNOSIS_EXECUTION.md)。',
        '- [拟合起点与留出起点的教师—学生演示](diagnosis_demo.mp4)。',
        f'- [全部新增原始数据、日志和校验清单]({release})；先前数据来自 v0.3.0-bc-pilot。',
        '- 仅使用GPU1，无LLM API。此处只有一个模型、一个任务的开发诊断，不能作为独立测试或发表结论。',
        '- 首次batch=8重建在教师seed1210000上误差0.0001975，超过预设1e-4，故中止。失败记录保留；经独立预检改用与现场一致的batch=1逐帧重建，容差、模型、数据和主指标不变，闭环回合没有重跑。详见[精度诊断与续算协议](../../docs/BC_DIAGNOSIS_SERIAL_RECOVERY.md)。',
        '- 本报告的4000步为20个正式回合的 env.step 控制步，独立预检另有8步；不是PhysX内部积分子步数。','',
        '|阶段|实际进程耗时|PyTorch峰值分配显存|','|---|---:|---:|']
for stage,name in [('witness','短预检'),('fit','20个拟合起点测试'),('serial_witness','串行重建独立预检'),('offline_serial','80条轨迹串行离线分析')]:
    process=load(BASE/f'process_{stage}.json')
    lines.append(f"|{name}|{process['wall_seconds']/60:.2f} 分钟|{process['peak_torch_mib']:.1f} MiB|")
lines+=['',f"首次离线失败另耗时{load(BASE/'process_offline.json')['wall_seconds']:.2f}秒；失败进程未写出PyTorch峰值。原预检加串行预检、原离线失败加串行完成阶段，分别仍在原300秒与900秒阶段预算内。",'',
        'PyTorch分配显存不包含仿真、渲染和CUDA上下文；整卡采样另见 resources.json，包含当时其他进程。','']
(OUT/'报告.md').write_text('\n'.join(lines),encoding='utf-8')
table=''.join(f"<tr><td>{r['seed']}</td><td>{int(r['ever_grasp'])}</td><td>{int(r['success_once'])}</td><td>{int(r['success_at_end'])}</td><td><a href='{html.escape(r['folder'])}/rollout.mp4'>完整视频</a></td></tr>" for r in fit)
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>行为克隆失败诊断</title>
<style>body{{font:16px/1.7 system-ui;background:#f5f7fb;color:#24364d;max-width:1100px;margin:36px auto;padding:0 24px}}.card{{background:white;padding:24px;border-radius:16px;margin:24px 0}}video,img{{width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}table{{width:100%;border-collapse:collapse}}</style>
<h1>训练过的起点，学生能完成吗？</h1><div class="card">固定原BC模型，新增20次拟合起点测试：曾抓取 <b>{sf['ever_grasp']}/20</b> · 曾成功 <b>{sf['success_once']}/20</b> · 结束成功 <b>{sf['success_at_end']}/20</b>。<br>原留出成绩：4/20、3/20、1/20。拟合起点经过教师成功筛选，两组并非无偏泛化比较。没有新训练或教师接管。<br><a href="reports/bc_diagnosis_v1/报告.md">完整报告</a> · <a href="{release}">GitHub 数据</a></div>
<div class="card"><video controls preload="metadata" src="reports/bc_diagnosis_v1/diagnosis_demo.mp4"></video></div><div class="card"><img src="reports/bc_diagnosis_v1/diagnosis.png"></div>
<div class="card"><h2>全部20个拟合起点测试</h2><table><tr><th>种子</th><th>抓取</th><th>曾成功</th><th>结束成功</th><th>视频</th></tr>{table}</table></div></html>'''
(ROOT/'行为克隆失败诊断.html').write_text(page,encoding='utf-8')
print('BC_DIAGNOSIS_REPORT_PASS',json.dumps(stats),flush=True)
