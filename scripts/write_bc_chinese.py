"""Build the Chinese experiment report and local video index from audited results."""
import html,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'reports/bc_v1_2'; base=root/'artifacts/bc_v1_2'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
summary=load(out/'summary.json'); audit=load(out/'audit.json'); assert audit['passed']
teacher=load(base/'collection/episodes.json')[80:]; student=load(base/'student/episodes.json')
metrics=load(base/'student/metrics.json'); sel=summary['selection']; result=summary['student_dev']
gate=summary['gate_passed']; manifest=load(base/'student/training_manifest.json')
version='v0.3.0-bc-pilot'; release=f'https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/{version}'
rows=['|策略及数据组|回合数|曾抓取|曾完整成功|结束完整成功|','|---|---:|---:|---:|---:|']
for label,key,n in [('教师：全部拟合起点','teacher_fit_all',80),('教师：开发留出起点','teacher_dev',20),('学生：开发留出起点','student_dev',20)]:
    c=summary[key]; rows.append(f"|{label}|{n}|{c['ever_grasp']}|{c['success_once']}|{c['success_at_end']}|")
lines=['# 训练集教师示范与行为克隆实验报告','',
       f"本轮完成 R001–R003：20 组教师接口配对、100 次固定任务示范、5,000 次行为克隆更新及 20 次闭环开发评估。工程门槛**{'通过' if gate else '未通过'}**：学生曾抓取 {result['ever_grasp']}/20、曾完整成功 {result['success_once']}/20、结束完整成功 {result['success_at_end']}/20。",'',
       '这是自己训练的状态输入行为克隆（BC）基线，当前没有新增世界模型或风险规划训练。本轮是固定 train 任务、单个训练种子的开发实验，不能作为跨布局、跨房屋、实机或完整仓储搬运的成绩。','',
       '## 1. 任务、流程和数据','',
       f"机器人为 Fetch 移动机械臂，任务为 ManiSkill-HAB TidyHouse Pick。固定布局 {sel['layout']}、task_idx {sel['task_idx']}、目标 `{sel['source_episode']['target_object']}`。选择依据为 R001 按编号排序的第一条教师结束完整成功回合，未根据开发成绩选择任务。",'',
       '流程：训练集官方 SAC 教师 → 同步采集深度/状态/动作 → 仅成功拟合轨迹训练 BC → 学生独立闭环执行 → 从逐步条件审计成功率。','',
       'R001 为 train 布局 6、22、37、52、68，每布局四组相同起点、两次 200 步，全部 20 组通过 1e-5 数值容差及布尔成功条件一致性。额外读取状态未在这些配对中改变教师结果；这是接口检查，不是任意中途状态恢复证明。','',
       '公开数据每任务有 100 个预生成起点。采集前用 Random(1210000) 的固定排列将编号 0–99 分成 80 个拟合、20 个开发留出起点，无放回采集，每个起点一次。教师采集种子 1210000–1210099；不同种子之外，还显式隔离起点编号。全部失败示范保留，旧版 val 轨迹没有进入训练。','',
       f"实际用于拟合的是前 80 起点中结束完整成功的 {summary['fit_successful_demos']} 条示范，共 {summary['fit_transitions']:,} 个状态—动作对。20 个留出起点不进入优化器或归一化统计；模型固定选第 5,000 次更新，不按留出成绩挑选。",'',
       '原始保存内容：初始模拟器/控制器快照、完整初始三帧深度、后续每步两路最新深度、201 个观测、200 个原始 13 维/执行 13 维/有效 11 维动作，以及奖励、接触量和五项完整成功条件。可重建教师每一步的三帧输入。','',
       '## 2. 学生如何训练与测试','',
       f"网络为 45→256→256→11，ReLU 隐层、tanh 输出，共 {manifest['parameters']:,} 个训练参数。45 维为原生 42 维状态加触力/100、累积触力/5000、步数/200；包含模拟器目标位姿，缺少完整障碍几何。字段顺序和单位见逐回合 schema.json。",'',
       '11 个有效动作对应机械臂、夹爪、升降躯干和底盘；两个头部控制通道固定为零。夹爪是绝对开合命令。监督目标严格为 state[t]→teacher_action[t]；损失为运动十维平均 MSE 加夹爪一维 MSE，等权相加。Adam，学习率 3e-4，batch 256，训练种子 1300001，共 5,000 更新。','',
       '测试时学生自行输出全部动作，没有教师辅助。20 个留出起点与教师对应回合使用相同种子、语义任务和 spawn_idx；活动物体、机器人、控制器与观测逐一核对，完整初始教师观测字节相同。仅规范化进程局部场景编号及万米外隐藏物体的 z 停放槽。每次完整运行 200 步，不在首次成功后提前停止。','',
       '完整成功保留原 HAB 五项联合条件：抓住物体、末端回到休止区域、机器人处于休止状态、满足静止要求、累计触力未超限。曾抓取、曾完整成功、结束完整成功分别报告。','',
       '## 3. 结果','',*rows,'','![训练与闭环结果](results.png)','',
       f"动作监督总损失：拟合 {metrics[0]['fit']['total']:.6f} → {metrics[-1]['fit']['total']:.6f}；开发 {metrics[0]['dev']['total']:.6f} → {metrics[-1]['dev']['total']:.6f}。第一条拟合轨迹：{metrics[0]['first_fit_trajectory']['total']:.6f} → {metrics[-1]['first_fit_trajectory']['total']:.6f}。动作拟合改善不能替代闭环成功。",'',
       '末步失败条件计数（同一回合可同时缺少多项，不能相加当失败回合数）：'+ '；'.join(f'{k}={v}' for k,v in summary['student_failed_end_conditions'].items())+'。','',
       '工程门槛预先固定为曾抓取≥10/20、曾完整成功≥5/20。'+('本轮达到这一受限任务门槛，后续须另行建立具有任务能力的示范辅助世界模型；BC 成功不等于世界模型成功。' if gate else '本轮未达到门槛，因此未启动世界模型/风险扩展。需结合拟合误差与闭环失败检查状态分布偏移、动作多模态和观测不足，不能只扩大从头 RL 步数。'),'',
       '## 4. 耗时、验证和边界','',
       '|阶段|实际进程耗时|PyTorch 峰值分配显存|','|---|---:|---:|']
for name,key in [('固定起点短预检','witness'),('R001 教师配对','interface'),('R002 示范采集','collect'),('R003 训练与评估','train')]:
    s=summary['stages'][key]; peak='未记录完整阶段峰值' if s['peak_torch_mib'] is None else f"{s['peak_torch_mib']:.1f} MiB"
    lines.append(f"|{name}|{s['wall_seconds']/60:.2f} 分钟|{peak}|")
lines += ['', '仅使用 GPU1。PyTorch 峰值不包括仿真、渲染和 CUDA 上下文；整卡采样统计与原始资源日志另存。阶段计划上限为 R001 1 小时、R002 2 小时、R003 1 小时，没有调用 LLM API。','',
          f"原始数据审计通过：{audit['formal_episodes']} 个正式回合、{audit['formal_steps']:,} 个物理步、{len(audit['video_counts'])} 段各 201 帧完整视频、20 对教师/学生起点、5,000 条更新日志；训练归一化来源隔离。最终模型重算学生动作的最大绝对差为 {audit['student_action_replay_max_abs_error']:.3g}。",'',
          '首次短预检因误用逐位相等而失败；原始记录和修正前源码均保留。后续改回原计划 1e-5 容差。起点编号分割在任何 R002 采集之前固定。训练后的第一个测试因全量起点哈希检查失败而暂停；诊断为进程局部编号与远端隐藏物体停放槽差异。教师 8 步前缀重放验证后，保留第一例学生失败，只续跑其余 19 例，模型没有再次更新。原失败进程日志保留，原训练阶段 PyTorch 峰值未写出，表中标为不可得；整卡资源日志仍存在。见[入口诊断](../BC入口诊断.md)和[续跑协议](../../docs/BC_EVALUATION_RECOVERY.md)。','',
          '仅训练一个模型种子、开发起点 20 个；样本较小，没有独立房屋测试。拟合只用成功教师轨迹，闭环偏离示范后可能缺乏纠错样本。即使监督误差很低，也不能据此称已理解接触动力学。','',
          '## 5. 代码、完整数据与演示','',
          '- [代码仓库](https://github.com/Pendragon08/hab-contact-aware-world-model)。',
          '- [固定起点执行协议](../../docs/BC_FIXED_SPAWN_EXECUTION.md)与[冻结配置](../../configs/bc_v1_2.json)。',
          f'- [完整数据、检查点、日志与校验清单]({release})。',
          '- [教师—学生对比演示](teacher_student_demo.mp4)：按种子顺序选第一例学生曾成功和第一例未曾成功，不以示例替代统计。',
          '- 本地打开根目录 `行为克隆实验汇总.html`，可以逐一播放全部 20 对教师/学生完整视频。',
          '- 下载 Release 资产至 release 后，用 restore_artifacts.py 校验并恢复；Git 本身不包含大文件。','']
(out/'报告.md').write_text('\n'.join(lines), encoding='utf-8')
table=[]
for i,(t,s) in enumerate(zip(teacher,student)):
    status=lambda r: f"抓取 {int(r['ever_grasp'])} / 曾成功 {int(r['success_once'])} / 结束成功 {int(r['success_at_end'])}"
    table.append(f"<tr><td>{i+1}</td><td>{s['seed']}</td><td>{s['spawn_idx']}</td><td>{status(t)}<br><a href='{html.escape(t['folder'])}/rollout.mp4'>教师完整视频</a></td><td>{status(s)}<br><a href='{html.escape(s['folder'])}/rollout.mp4'>学生完整视频</a></td></tr>")
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>行为克隆实验汇总</title>
<style>body{{font:16px/1.7 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#203047;background:#f5f7fb}}h1{{font-size:32px}}.card{{background:white;padding:24px;border-radius:16px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dce2ed;padding:12px;text-align:left}}video,img{{width:100%;border-radius:12px}}a{{color:#155ec2}}.lead{{font-size:21px}}</style>
<h1>自己的抓取策略：教师示范 → 行为克隆</h1><p class="lead">工程门槛：{'通过' if gate else '未通过'} · 学生曾抓取 {result['ever_grasp']}/20 · 曾完整成功 {result['success_once']}/20 · 结束完整成功 {result['success_at_end']}/20</p>
<div class="card">固定 train 布局 {sel['layout']}，Fetch 移动机械臂，100 个公开预生成起点按 80/20 隔离。使用 {summary['fit_successful_demos']} 条成功拟合示范，5,000 次更新。所有评估完整运行 200 步。<br><b>这是状态输入 BC 开发实验；世界模型和风险规划尚未在本轮训练。</b><br><a href="reports/bc_v1_2/报告.md">完整报告</a> · <a href="{release}">GitHub 完整归档</a></div>
<div class="card"><h2>教师与学生，同一起点</h2><p>示例按固定规则选择，包含可用的第一例成功和第一例失败。下面的表格提供全部 20 对视频。</p><video controls preload="metadata" src="reports/bc_v1_2/teacher_student_demo.mp4"></video></div>
<div class="card"><img src="reports/bc_v1_2/results.png" alt="训练损失与闭环计数"></div>
<div class="card"><h2>全部开发回合</h2><table><tr><th>序号</th><th>种子</th><th>起点编号</th><th>教师</th><th>训练后的学生</th></tr>{''.join(table)}</table></div></html>'''
(root/'行为克隆实验汇总.html').write_text(page,encoding='utf-8')
print('CHINESE_BC_REPORT_COMPLETE', result, gate)
