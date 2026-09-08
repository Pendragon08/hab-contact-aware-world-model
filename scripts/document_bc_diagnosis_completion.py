"""Publish the completed diagnosis in project indexes without rewriting the frozen protocol."""
import datetime as dt,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
load=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
s=load('reports/bc_diagnosis_v1/summary.json');a=load('reports/bc_diagnosis_v1/audit.json')
inv=load('release/bc-diagnosis-v1.release.json');assert a['passed']
g=s['groups']['student_fit'];d=s['groups']['student_dev']
stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
release='https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.3.1-bc-diagnosis'
section=f'''## 最新：固定策略闭环失败诊断

原 BC 模型保持第5000次更新，新测20个真正参与训练的起点：曾抓取 **{g['ever_grasp']}/20**、曾完整成功 **{g['success_once']}/20**、结束完整成功 **{g['success_at_end']}/20**。原留出测试仍为4/20、3/20、1/20，没有重跑或用于新训练。

本轮新增4000个物理步，分析80条轨迹的教师建议与学生动作，并计算相对拟合状态的距离。拟合起点来自成功教师示范，两组不构成无偏泛化比较；动作分歧和距离是诊断证据，尚未隔离唯一原因。未进行参数更新、世界模型或风险规划训练，原R003门槛仍未通过。

- [诊断报告](reports/bc_diagnosis_v1/报告.md) · [冻结协议](docs/BC_DIAGNOSIS_EXECUTION.md) · [独立预检](reports/BC诊断入口验证.md)
- [完整新增原始数据]({release}) · [训练起点与留出起点演示]({release.replace('/tag/','/download/')}/diagnosis_demo.mp4)
- 本地打开 `行为克隆失败诊断.html` 查看全部20条新视频；[逐文件交付校验](reports/bc_diagnosis_local_transfer_audit.json)与[发布校验](reports/github_v0.3.1-bc-diagnosis_audit.json)。

'''
p=root/'README.md';text=p.read_text(encoding='utf-8')
assert '## 最新：固定策略闭环失败诊断' not in text
offset=text.index('[计划与执行结果]')
p.write_text(text[:offset]+section+text[offset:],encoding='utf-8')
ledger=f'''# 行为克隆失败诊断执行记录

日期：{stamp}，北京时间。固定协议提交7007629，模型与此前训练配置均未改变。

|阶段|实际完成|结果|
|---|---|---|
|D001 预检|原学生起点8步，加旁路教师和离线重建|独立预检通过；学生前缀所有数组差为0|
|D002 训练起点闭环|20回合、4000步，0更新|抓取{g['ever_grasp']}/20，曾成功{g['success_once']}/20，结束成功{g['success_at_end']}/20|
|D003 离线诊断|80轨迹、16000个动作时点|教师重建最大误差{a['teacher_replay_max_error']:.3g}；留出标签不训练|

原20次留出成绩仍为4/20抓取、3/20曾成功、1/20结束成功；拟合组成绩不能替代原留出门槛。因此R004/R005仍GATED。本轮没有新的优化器更新。

拟合组按教师成功筛选，不能将它与全部留出组直接解释为无偏泛化差。原教师轨迹上的BC动作MSE为{s['groups']['teacher_fit']['total_mse_mean']:.5f}，学生访问拟合轨迹上为{g['total_mse_mean']:.5f}。该差异并未单独隔离观测不足、动作多解或分布偏移。

下一轮优先比较：同模型输入、相同更新预算下，加入训练起点上的学生访问状态教师标签，或等量原教师状态样本；只使用训练分组。此训练本轮未启动。随后再单独考虑增加短历史、底盘速度或深度输入。

原始数据包含{inv['files']}文件，压缩{inv['archive_bytes']/2**20:.1f}MiB；代码、配置、报告与完整原始数据均归档。详见[报告](../reports/bc_diagnosis_v1/报告.md)和[版本数据]({release})。
'''
(root/f'refine-logs/BC_DIAGNOSIS_{stamp}.md').write_text(ledger,encoding='utf-8')
with (root/'refine-logs/EXPERIMENT_TRACKER.md').open('a',encoding='utf-8') as f:
    f.write(f'\n## 后续诊断批次已完成（{stamp}）\n\n新增20个训练起点测试及80条离线诊断；0参数更新。训练起点抓取{g["ever_grasp"]}/20、曾成功{g["success_once"]}/20、结束成功{g["success_at_end"]}/20。原R003留出门槛仍未通过，R004/R005不启动。详见[执行记录](BC_DIAGNOSIS_{stamp}.md)。\n')
with (root/'MANIFEST.md').open('a',encoding='utf-8') as f:
    f.write(f'| {stamp} +08:00 | run-experiment | refine-logs/BC_DIAGNOSIS_{stamp}.md | completed | 固定BC、20个拟合起点、80轨迹诊断、0更新 |\n')
notes=f'''固定BC闭环失败诊断完成：新增20个已拟合起点测试、4000物理步；对80条原始/学生轨迹计算冻结教师建议与状态偏离。模型没有更新，原20次留出结果没有重试。

拟合起点：曾抓取{g['ever_grasp']}/20，曾完整成功{g['success_once']}/20，结束成功{g['success_at_end']}/20。原留出仍为4/20、3/20、1/20。拟合起点按教师成功筛选，不是无偏泛化比较。原门槛未通过，没有进入世界模型或风险规划。

本版本归档{inv['files']}个文件，包括完整新视频、状态/深度/快照、逐步旁路教师建议、80条离线分析数组、日志及预检。依赖的原模型与原始教师/留出轨迹从v0.3.0-bc-pilot恢复；本版本只打包新增数据。

将全部资产下载到release目录后：
```bash
python scripts/restore_artifacts.py --inventory release/bc-diagnosis-v1.release.json --root .
```
打开根目录“行为克隆失败诊断.html”查看演示和所有20条新视频。完整报告位于reports/bc_diagnosis_v1/报告.md。GPU1 only，无LLM API。
'''
(root/'reports/bc_diagnosis_v1/release_notes.md').write_text(notes,encoding='utf-8')
print('BC_DIAGNOSIS_DOCUMENTATION_COMPLETE',stamp)
