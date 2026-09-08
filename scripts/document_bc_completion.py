"""Update the project index and versioned execution ledger after audited BC completion."""
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8'))
s = load('reports/bc_v1_2/summary.json')
a = load('reports/bc_v1_2/audit.json')
resources = load('reports/bc_resources.json')
inventory = load('release/bc-pilot-v1.release.json')
assert a['passed'] and a['formal_episodes'] == 160 and a['formal_steps'] == 32000
c = s['student_dev']; t = s['teacher_dev']; gate = s['gate_passed']
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
stamp = now.strftime('%Y%m%d_%H%M%S'); display = now.isoformat(timespec='seconds')
release = 'https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.3.0-bc-pilot'
decision = ('达到受限任务门槛；R004 仍需预先固定预算、存储和执行协议。' if gate else
            '未达到预定门槛（曾抓取≥10/20 且曾完整成功≥5/20），R004/R005 保持 GATED。下一步先诊断状态输入 BC 的闭环失败。')
completion = f'''## 本批执行结果（{display}）

已执行用户授权的有限批次 R001→R002→R003。原始计划保存在 `EXPERIMENT_PLAN_20260909_001000.md`，预算与门槛未根据测试成绩修改。

|运行|实际完成|实测结果|实际进程耗时|决策|
|---|---|---|---:|---|
|R001|20 组、40 回合、8,000 步|20/20 配对通过；最大轨迹差 0|{s['stages']['interface']['wall_seconds']/60:.2f} 分钟|PASS|
|R002|100 回合、20,000 步|80/20 不重复公开起点；53 条成功拟合示范、10,600 样本|{s['stages']['collect']['wall_seconds']/60:.2f} 分钟|PASS|
|R003|5,000 更新、20 回合、4,000 步|学生抓取 {c['ever_grasp']}/20、曾成功 {c['success_once']}/20、结束成功 {c['success_at_end']}/20|{s['stages']['train']['wall_seconds']/60:.2f} 分钟|{'PASS' if gate else 'COMPLETE / GATE_FAILED'}|

{decision}

同起点教师：抓取 {t['ever_grasp']}/20、曾成功 {t['success_once']}/20、结束成功 {t['success_at_end']}/20。教师用三帧深度，学生只用 45 维状态；输入信息不同，不能将差距全部归因于学习算法。

采集前配置 `configs/bc_v1_2.json` SHA256：`7c2dc173dae9ecd06eda6a95d3cbe27079879c045a166993d163469cbb4483a6`。训练仅使用成功拟合轨迹，留出起点未参与优化器和归一化；固定使用第 5,000 次更新模型，没有按留出结果挑选检查点。

R003 原训练进程在首例测试完成后因严格全量起点哈希检查暂停。活动状态与完整初始观测相同，差异来自进程局部场景索引和万米外隐藏物体停放槽；独立教师 8 步重放全字段误差为 0 后，保留首例学生失败，只续跑剩余 19 例。模型没有重新训练。原始失败日志及代码均保留，见 `../docs/BC_EVALUATION_RECOVERY.md`。原训练峰值显存未写出，不伪造该数值。

R001 汇总控制台日志曾因运行中的启动文件被修改而截断，原始逐步日志、轨迹和视频完整，重新核验全部 8,000 步通过。该事件和最初容差预检失败见 `../reports/BC入口诊断.md`。上游静止判据的有符号速度问题作为附加诊断报告，保留原始 HAB 主指标和训练筛选。

完整审计：160 个正式回合、32,000 物理步、160 段 201 帧视频，5,000 次真实更新；20 对语义起点一致；模型重算动作最大误差 {a['student_action_replay_max_abs_error']:.3g}。仅 GPU1 执行，无 LLM API。

本地报告 `../reports/bc_v1_2/报告.md`，演示入口 `../行为克隆实验汇总.html`；[GitHub 数据归档]({release}) 包含 {inventory['files']} 个文件，原始字节 {inventory['uncompressed_bytes']:,}，压缩字节 {inventory['archive_bytes']:,}。上传和本地恢复的独立校验另存 reports。
'''
plan = (ROOT/'refine-logs/EXPERIMENT_PLAN_20260909_001000.md').read_text(encoding='utf-8')
plan = plan.replace('**状态**：建议方案，尚未实现或启动本轮训练。所有数字门槛均为拟定的工程决策标准，不是已有成绩或文献标准。',
                    '**状态**：首批 R001–R003 已执行；实测结果和后续门槛见下方更新。原计划中的“拟定”数字是预先设定的工程决策标准。')
plan = plan.replace('## 1. 已有证据与当前决策', completion+'\n## 1. 已有证据与当前决策', 1)
tracker = '# 实验执行表\n\n'+completion+f'''
|后续运行|内容|当前状态|进入条件|
|---|---|---|---|
|R004|示范辅助世界模型与匹配 SAC|GATED|R003 可学习性门槛通过，预算/存储/协议明确|
|R005|动作相关接触预测与风险规划|GATED|世界模型已具备任务能力，正式比较与消融冻结|
|R006|Place / Pick→Place 扩展|DEFERRED|核心研究主张有证据|

本轮只完成一个训练种子、一个 train 任务的开发评估；不是独立最终 benchmark，也没有新增世界模型或风险规划结果。
'''
for name, content in [('EXPERIMENT_PLAN',plan),('EXPERIMENT_TRACKER',tracker)]:
    content = '\n'.join(line.rstrip() for line in content.splitlines()) + '\n'
    (ROOT/f'refine-logs/{name}_{stamp}.md').write_text(content,encoding='utf-8')
    (ROOT/f'refine-logs/{name}.md').write_text(content,encoding='utf-8')

readme = (ROOT/'README.md').read_text(encoding='utf-8')
readme = readme.replace('**当前完成：300 回合接触控制配对实验，以及 TD-MPC2 / 状态 SAC 各 10,000 步短程训练。风险预测模块尚未训练，也未得到世界模型提升 HAB 性能的研究结论。**',
                        '**当前完成：接触控制配对实验、TD-MPC2 / 状态 SAC 短程训练，以及教师示范和行为克隆实验。行为克隆未通过可学习性门槛；风险预测模块尚未训练，也未得到世界模型提升 HAB 性能的研究结论。**')
start = readme.index('[下一轮计划与验收门槛]')
end = readme.index('\n公开数据中每个训练任务',start)
readme = readme[:start]+f'''[计划与执行结果](refine-logs/EXPERIMENT_PLAN.md) · [执行表](refine-logs/EXPERIMENT_TRACKER.md)：**R001–R003 已完成**。20/20 教师接口配对通过，采集 100 次示范，用 53 条成功拟合轨迹完成 5,000 次 BC 更新。学生在 20 个留出起点上曾抓取 **{c['ever_grasp']}/20**、曾完整成功 **{c['success_once']}/20**、结束完整成功 **{c['success_at_end']}/20**。{decision}

- [行为克隆报告](reports/bc_v1_2/报告.md) · [观测与静止判据诊断](reports/bc_v1_2/观测与静止判据诊断.md)
- [本轮完整数据、所有检查点和失败记录]({release}) · [教师—学生 Demo]({release.replace('/tag/','/download/')}/teacher_student_demo.mp4)
- 本地恢复数据后打开 `行为克隆实验汇总.html`，可播放全部 20 对同起点视频。
- [固定起点学习协议](docs/BC_FIXED_SPAWN_EXECUTION.md) · [评估恢复协议](docs/BC_EVALUATION_RECOVERY.md) · [入口与日志异常记录](reports/BC入口诊断.md)

这是 45 维状态行为克隆的开发实验。教师的留出曾成功成绩为 {t['success_once']}/20，使用深度历史输入；两者并非匹配观测的算法比较。本轮仍未证明世界模型或风险规划有效。
''' + readme[end:]
readme = readme.replace('本轮归档共 3,113 个文件', 'v0.2 归档共 3,113 个文件')
bc_archive = f"\nv0.3 行为克隆归档共 {inventory['files']} 个文件，压缩包 {inventory['archive_bytes']/2**20:.1f} MiB；[本地逐文件恢复审计](reports/bc_local_transfer_audit.json)与[GitHub 发布审计](reports/github_v0.3.0-bc-pilot_audit.json)提供完整性记录。[本轮资源采样](reports/bc_resources.json)：GPU1 整卡最高 {resources['gpu1']['max_used_mib']} MiB，GPU0 最低空闲 {resources['gpu0']['min_free_mib']} MiB；包含当时所有进程，5 秒采样不能保证捕获瞬时峰值。\n"
readme = readme.replace('## 代码、数据和论文来源',bc_archive+'\n## 代码、数据和论文来源',1)
(ROOT/'README.md').write_text(readme,encoding='utf-8')
with (ROOT/'MANIFEST.md').open('a',encoding='utf-8') as f:
    for name in ['EXPERIMENT_PLAN','EXPERIMENT_TRACKER']:
        f.write(f'| {display} | run-experiment | refine-logs/{name}_{stamp}.md | completed | R001–R003 实测结果、故障留痕与后续门槛；固定文件同步更新 |\n')
    f.write(f'| {display} | run-experiment | reports/bc_v1_2/报告.md | completed | 160 正式回合、5000 更新、20 起点开发评估与 Demo |\n')
notes = f'''R001–R003 有限批次完成：教师接口 20/20 配对通过，100 次教师尝试，53 条成功拟合示范，5,000 次 BC 更新和 20 次完整闭环评估。

学生：曾抓取 {c['ever_grasp']}/20，曾完整成功 {c['success_once']}/20，结束完整成功 {c['success_at_end']}/20。{decision}

同起点教师：曾完整成功 {t['success_once']}/20，结束完整成功 {t['success_at_end']}/20。教师使用深度历史，学生使用状态输入；这是单任务开发实验，不是世界模型或风险规划成果。

归档 {inventory['files']} 个文件，保留全部原始轨迹、深度、快照、视频、每 250 更新检查点、日志、失败尝试和源码快照。160 正式回合 / 32,000 步审计通过；初始状态规范化恢复保留首例失败，仅续跑19例。上游静止判据附加诊断没有改变主指标。

下载全部资产到 release/ 后执行：

```bash
python scripts/restore_artifacts.py --inventory release/bc-pilot-v1.release.json --root .
```

本地打开“行为克隆实验汇总.html”查看全部20对视频。压缩包和逐文件 SHA256 清单一并提供。仅 GPU1 执行，无 LLM API。
'''
(ROOT/'reports/bc_v1_2/release_notes.md').write_text(notes,encoding='utf-8')
print('BC_DOCUMENTATION_UPDATED',stamp)
