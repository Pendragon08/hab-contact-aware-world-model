"""Explain observed upstream metric semantics without altering experimental outcomes."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'reports/bc_v1_2'
d=json.loads((root/'reports/bc_semantics_audit.json').read_text())
lines=['# 观测与静止判据的补充诊断','',
       '本文件是原始结果之外的解释性检查。主报告的 HAB 指标、BC 拟合样本、固定训练步骤和工程门槛均保持冻结，没有根据这里的检查重新筛选示范或修改成绩。','',
       '## 1. 45 维输入不是完整物理状态','',
       '当前上游 `_get_obs_agent()` 将 qpos 与 qvel 的前三项底盘自由度去除。45 维学生输入由余下 12 维关节位置、12 维关节速度、TCP/物体相对位姿、目标位置、抓取标志，以及触力、累积触力、时间构成。它没有显式底盘速度，也没有深度帧历史。官方教师除了 42 维状态还使用三帧头部与手部深度图。','',
       '因此，学生和教师的输入信息并不相同，不能把二者的性能差异都归因于网络或优化算法。状态输入虽含模拟器目标位姿，也不等于完整马尔可夫状态。闭环失败可能与信息不足、分布偏移或示范数量有关，本诊断没有分离各自贡献。','',
       '后续若增加底盘速度、短历史或深度，应建立新版本，并给全部学习对照相同信息；本轮没有中途增加这些输入。','',
       '## 2. 固定上游版本的 is_static 使用带符号速度比较','',
       '在本项目固定的 ManiSkill 提交 `17121e3f96e3ee3ed0c03610b17f8bc2864617af` 中，Fetch.is_static 对躯干/手臂与底盘速度分别比较 `qvel <= threshold`，没有取绝对值；HAB Pick 的成功判据调用此函数。因此某些较大的负速度也可能通过这一布尔判断，名称为“静止”不能直接理解为所有绝对速度都很小。','',
       '这里从原始物理状态记录独立重算上游判断，并与每一步 is_static 对照一致；再作辅助检查，将同一批躯干/手臂速度绝对值限制为 0.2、底盘三个速度绝对值限制为 0.05，保留其余四项成功条件。单位随转动/移动关节不同，为 rad/s 或 m/s。没有重新运行或改变物理过程。','',
       '|组别|回合数|原判据曾成功|绝对速度辅助判据曾成功|原判据结束成功|绝对速度辅助判据结束成功|','|---|---:|---:|---:|---:|---:|']
names={'interface_teacher':'R001 教师（20 个不同任务起点）','teacher_fit_all':'R002 教师拟合起点（全部）','teacher_dev':'R002 教师留出起点','student_dev':'R003 学生留出起点'}
for key,g in d['groups'].items():
    lines.append(f"|{names.get(key,key)}|{g['episodes']}|{g['original_success_once']}|{g['absolute_speed_success_once']}|{g['original_success_end']}|{g['absolute_speed_success_end']}|")
lines += ['', '这两套指标不能混用。主报告及门槛沿用原始 HAB 规则，辅助判据只用于判断演示末端是否具有更严格的低速状态。两者都不是完整机器人安全保证。后续正式实验应在协议冻结时同时明确报告原始 benchmark 与物理意义更清楚的补充诊断，不能在实验后替换主指标。','',
          '原始逐回合计算记录见 [bc_semantics_audit.json](../bc_semantics_audit.json)。原始控制台日志异常与数据复核见[入口诊断](../BC入口诊断.md)。','']
(out/'观测与静止判据诊断.md').write_text('\n'.join(lines),encoding='utf-8')
p=out/'报告.md'; text=p.read_text(encoding='utf-8')
text+='\n## 6. 补充语义检查\n\n45 维输入没有显式底盘速度或深度历史；固定上游版本的 is_static 使用带符号速度比较，不能直接解释为所有绝对速度均很小。主指标、训练样本和门槛保持原 HAB 规则；额外计算的绝对速度结果见[观测与静止判据诊断](观测与静止判据诊断.md)。日志收尾异常导致 R001 控制台汇总文本丢失，但 40 个回合逐步原始记录均保留且独立复核通过。\n'
p.write_text(text,encoding='utf-8')
print('BC_SEMANTICS_REPORT_COMPLETE')
