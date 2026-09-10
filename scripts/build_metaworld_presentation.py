"""Build a Chinese report source and reproducible plots from saved evidence."""
from pathlib import Path
import csv, hashlib, io, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/metaworld_presentation'
OUT.mkdir(parents=True, exist_ok=True)
evidence_path = ROOT / 'hab_tdmpc2_original/reports/verified-evidence.json'
evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
rows = evidence['mw_evaluations']
done = evidence['tdmpc2_original/artifacts/seed1_500k/complete.json']
demo_path = ROOT / 'tdmpc2_original/artifacts/demo/final_demo_seed101_103/summary.json'
demo = json.loads(demo_path.read_text(encoding='utf-8'))
assert [r['train_native_steps'] for r in rows] == list(range(0, 500001, 100000))
assert [round(r['episode_success'] * 10) for r in rows] == [0, 0, 6, 9, 9, 8]
assert done['train_native_steps'] == 500000 and done['training_seed'] == 1
assert demo['optimizer_updates'] == 0 and demo['checkpoint_unchanged']

with (OUT / 'stage_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['train_native_steps', 'train_decisions', 'success_count', 'eval_episodes', 'success_rate', 'mean_episode_return'])
    for r in rows:
        writer.writerow([r['train_native_steps'],r['train_decisions'],round(r['episode_success']*10),10,r['episode_success'],r['episode_reward']])
(OUT / 'verified_results.json').write_text(json.dumps({'source':evidence_path.relative_to(ROOT).as_posix(), 'source_sha256':hashlib.sha256(evidence_path.read_bytes()).hexdigest(), 'complete':done, 'stages':rows, 'supplementary_demo':demo},ensure_ascii=False,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
x=[r['train_native_steps']/1000 for r in rows]
for ax, y, color, title, ylabel in [(axes[0],[r['episode_success']*100 for r in rows],'#136c60','Terminal success','Success rate (%)'),(axes[1],[r['episode_reward'] for r in rows],'#205b9f','Mean episode return','Return')]:
    ax.plot(x,y,marker='o',lw=2.4,color=color)
    ax.set(xlabel='Training environment steps (thousands)',ylabel=ylabel,title=title,xticks=x)
    ax.grid(alpha=.18)
axes[0].set_ylim(-5,105)
for xx,r in zip(x,rows):
    axes[0].annotate(f"{round(r['episode_success']*10)}/10",(xx,r['episode_success']*100),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
axes[1].set_ylim(0,1700)
fig.savefig(OUT/'learning_curves.svg')
fig.savefig(OUT/'learning_curves.png',dpi=180)
plt.close(fig)
svg=(OUT/'learning_curves.svg').read_text(encoding='utf-8')
svg=svg[svg.index('<svg'):]
svg=re.sub(r'width="[^"]+"', 'width="100%"', svg,count=1)
svg=re.sub(r'height="[^"]+"', '',svg,count=1)
svg=re.sub(r'\s+', ' ',svg)
chart=f'<figure style="margin:16px 0;padding:12px;background:white;border:1px solid #d4dce6;border-radius:10px">{svg}<figcaption>左：末步成功率；右：平均回合奖励。每点10回合，单个训练种子。连线仅连接阶段观测，不表示中间步数经过评估；未绘制跨种子置信区间。</figcaption></figure>'

table='\n'.join(f"| {r['train_native_steps']:,} | {round(r['episode_success']*10)}/10 | {r['episode_success']:.0%} | {r['episode_reward']:,.2f} |" for r in rows)
video_specs=[('补录成功 · 最终模型 · 种子101','tdmpc2_original/artifacts/demo/final_demo_seed101_103/seed_101.mp4','冻结最终权重的补充演示；不计入正式8/10。'),('正式评估失败 · 最终模型','tdmpc2_original/artifacts/demo/eval_250000.mp4','最终正式评估的首回合，属于2个失败回合之一。'),('训练前 · 随机初始化','tdmpc2_original/artifacts/demo/eval_000000.mp4','训练前评估首回合，用于展示尚未学会任务时的行为。')]
videos=[]
for title, rel, desc in video_specs:
    assert (ROOT/rel).is_file(), rel
    videos.append(f'<figure style="margin:18px 0;padding:16px;background:#fff;border:1px solid #d4dce6;border-radius:10px"><figcaption><strong>{title}</strong> — {desc}</figcaption><video controls playsinline preload="metadata" style="width:100%;max-height:410px;display:block;margin-top:12px;background:#14222b;border-radius:8px" src="{rel}">浏览器不支持视频，请通过演示包播放。</video></figure>')

md='''# Meta-World：TD-MPC2复现与实验结果

**汇报定位：原方法单任务、缩减预算复现。** 实验完成于2026年9月9日；本报告整理于2026年9月10日，依据已保存的完成记录、阶段指标和演示证据。本文报告TD-MPC2，而非2022年的第一代TD-MPC。

> 💡 主要结果：从随机权重训练50万原生环境步，最终末步成功8/10（80%），中途最好9/10（90%）；总运行约4小时4分钟。已获得有成功表现的世界模型＋MPC基线，尚未验证自研算法改进。

## 01 · 这次具体复现了什么

| 项目 | 本轮实际设置 |
|---|---|
| 论文 | Hansen、Su、Wang，TD-MPC2: Scalable, Robust World Models for Continuous Control，ICLR 2024 |
| 仿真任务 | Meta-World Pick-Place V2；pick-place-v2-goal-observable |
| 机器人与目标 | Sawyer机械臂抓取物体，并移到目标位置 |
| 输入与动作 | 39维状态；4维动作：末端三方向移动＋夹爪 |
| 模型 | 官方约5M参数配置，潜在表示512维，5个Q网络 |
| 初始化与数据 | 随机权重；自行仿真交互，无教师、外部预训练权重或专家示范 |
| 训练预算 | 250,000次决策 × 动作重复2次 = 500,000个原生环境步 |
| 单回合长度 | 100次决策，对应200个原生环境步（reset附加步另计） |
| 训练种子与GPU | seed=1；GPU1，RTX A6000 48GB |
| 用时 | 14,631.65秒，约4小时4分钟；包含阶段评估 |

目标位置可以位于空中，成功不一定要求松开夹爪、把物体放回桌面。本轮读取仿真提供的状态向量，**视频用于展示，不是模型的视觉输入**。无需LLM API。

原论文：[PDF](https://arxiv.org/pdf/2310.16828v2) · [官方项目](https://www.tdmpc2.com/) · [官方代码固定版本](https://github.com/nicklashansen/tdmpc2/tree/57158282b46ebc5c329c5be9cfe2b0094126d1ca)。

## 02 · 算法在任务中怎样运行

<div style="padding:20px;background:#e9f2fb;border:1px solid #c7d9ed;border-radius:10px;line-height:2"><strong>39维环境状态 → 编码为512维潜在状态 → 生成候选动作序列 → 世界模型预测状态与奖励 → Q网络补充长期收益 → MPC迭代搜索 → 执行首动作（重复2次） → 新状态 → 再次规划。</strong><br>训练时另有学习闭环：实际轨迹 → 经验回放池 → 抽取片段 → 更新网络。</div>

世界模型回答“执行这个动作之后会怎样”；策略先验提供一部分候选；规划器比较预测回报并选择动作。原版规划在网络内部推演，不会把每条候选都放进MuJoCo实际执行。

| 规划参数 | 本轮数值与含义 |
|---|---|
| 规划时域 | 3次动作决策 |
| 候选总数 | 512条，**其中**24条由策略先验产生 |
| 精英数量 | 保留64条高分候选 |
| 搜索轮数 | 6轮，按收益权重更新采样分布 |
| 折扣系数 | 0.95 |
| 动作执行 | 从最终精英按权重抽取序列，执行首动作，再重新规划 |

评分由三步预测奖励和之后的Q价值组成。Q估计长期回报，不是成功概率。短时域不意味着只考虑三步以内的好处。评估关闭额外探索噪声，但候选采样与精英抽取仍有随机性。

详细算法教学见[TD-MPC2算法流程报告](TD-MPC2算法流程汇报.html)；实验接口见[runner.py](tdmpc2_original/runner.py)。

## 03 · 如何训练、怎样评估

训练先收集初始随机经验；seed_steps=1000，原循环的判断使索引0至1000共1,001次动作决策仍使用随机动作。在第1000步进行1,000次初始回放更新，此后交替规划、交互和学习。原日志中的“Pretraining agent on seed data”指随机初始经验学习，**不是专家示范预训练**。

每批256个连续片段，训练潜在表示一致性、奖励预测、Q价值与策略先验。使用官方损失和优化方式，没有加入行为克隆、接触风险头或动作后果排序模块。

| 评估规则 | 本轮执行方式 |
|---|---|
| 检查点 | 0、10、20、30、40、50万原生训练环境步 |
| 每轮回合数 | 10个完整回合；六轮共60个回合 |
| 参数与数据 | 冻结权重，评估轨迹不进入训练回放 |
| 主要指标 | 最后一个原生环境步的info.success，取10回合均值 |
| 辅助指标 | 平均回合累计环境奖励；动作重复的奖励相加 |
| 独立计数 | 训练50万原生步；评估6,000次决策；reset额外2,560个原生步 |

本轮验证的是**末步成功**，不是“中途曾经成功”，也不保证全回合一直保持成功。环境原生步也不等同于MuJoCo内部最细的积分子步。

## 04 · 六个阶段的实测结果

| 训练原生环境步 | 末步成功数 | 成功率 | 平均回合奖励 |
|---:|---:|---:|---:|
__TABLE__

__CHART__

数据源为保存的metrics数值记录，图表由脚本直接生成。原始表可下载[阶段CSV](reports/metaworld_presentation/stage_results.csv)，核验记录见[结果JSON](reports/metaworld_presentation/verified_results.json)，曲线可单独导出[SVG](reports/metaworld_presentation/learning_curves.svg)或[PNG](reports/metaworld_presentation/learning_curves.png)。

## 05 · 这些结果说明什么

1. **已学到可执行的控制行为。**正式评估从初始0/10增加到最终8/10，配套保存了实际执行视频。成功不是仅由预测奖励推断出来的。
2. **最终结果是80%，不能写成90%。**90%是30万与40万步的阶段最好值；50万步检查点为80%。
3. **表现并非单调改善。**最后一轮比最好阶段少成功1个回合，每轮只有10回合，不能据此断言出现过拟合、灾难性遗忘或训练退化。
4. **这是可用的初步基线证据。**只有一个训练种子、一个任务；尚不能说明稳定达到原论文水平、具有跨任务泛化能力或优于其他算法。

这里不绘制原论文数字的“提升百分比”：预算、种子数量与评估协议没有完全对齐，不能强行横向排名。也不将旧Meta-World V3示范辅助实验和本轮V2成绩直接相减作为方法收益。

## 06 · 成功演示与失败案例

正式评估的最终10回合仍是8成功、2失败。原程序每轮只自动录制首回合，最终轮首回合恰好失败，因此后来使用**冻结最终权重**，按预先固定的101、102、103三个种子补录，三次均成功，优化器更新为0。补录不加入正式8/10，也不是新的大样本评估。

__VIDEOS__

另外两段补录：[种子102](tdmpc2_original/artifacts/demo/final_demo_seed101_103/seed_102.mp4) · [种子103](tdmpc2_original/artifacts/demo/final_demo_seed101_103/seed_103.mp4)。视频按15 FPS导出，每段约6.7秒；播放时长不等于仿真物理时长。支持浏览器原生播放控件，不自动播放。

## 07 · 原论文复现与工程适配的边界

| 内容 | 本轮情况 | 能否称为自研算法创新 |
|---|---|---|
| 世界模型、SimNorm、奖励与Q、策略先验、MPC | 沿用官方固定版本 | 否 |
| 单任务、单种子、50万原生步 | 缩减实验范围与预算 | 否 |
| 精确预算终止与最后一次显式评估 | 避免额外一次决策，记录清楚终点 | 工程适配 |
| 本地日志、轨迹、视频、检查点哈希 | 保存可复查证据 | 工程完善 |
| Python/PyTorch/TorchRL兼容与MuJoCo库加载 | 适配现有服务器 | 工程适配，运行依赖并非逐项与论文相同 |
| 失败预测、接触约束排序、新的风险模块 | 本轮未加入 | 暂无算法改进结论 |

**目前最准确的名称：TD-MPC2在Meta-World Pick-Place V2上的单任务、缩减预算复现。**不是全论文复现，也不是已经完成新算法验证。

## 08 · 可复查性与数据保存情况

核心TD-MPC2版本：`57158282b46ebc5c329c5be9cfe2b0094126d1ca`。Meta-World固定版本：`04be337a12305e393c0caf0cbf5ec7755c7c8feb`。运行使用Python3.10、Torch2.5.1、TorchRL0.6、MuJoCo2.1系列，依赖差异见[环境规格](tdmpc2_original/configs/env-spec.json)。

本次报告依据的本地完成证据记录：2,500条训练轨迹、60条阶段评估轨迹，以及独立补录记录。最终权重补录前后哈希一致，详见[补录核验JSON](tdmpc2_original/artifacts/demo/final_demo_seed101_103/summary.json)。

按2026年9月9日保存的归档记录，服务器完整训练归档为222,474,625字节、2,596个文件；该记录显示全量包尚未完整回传或上传Release。**本次制作报告没有重新连接服务器核验归档状态，不声称已完成全量外部备份。**本报告便携包只包含汇报页面、曲线、结果证据、必要代码与演示媒体，不包含训练权重、全部轨迹或第三方仿真资产。

## 09 · 下一步实验与仍需解决的问题

- **先提升证据强度：**增加独立训练种子、扩大冻结评估集，报告种子间波动与成功率区间。
- **诊断失败来源：**比较抓取、目标到达及最终保持的轨迹表现；目前不能从两个失败回合推断唯一原因。
- **检查候选是否有改善空间：**在同一状态比较原选择和替代首动作的真实终局结果，先确认存在更好的候选。
- **验证新的评分模块：**只有候选覆盖门槛通过，才训练结果评分器，并与同数据普通critic、预测表示线性探针及原版MPC公平比较。

以上是研究计划，不是本轮已经完成的实验。Meta-World任务的原生成功定义与HAB接触力约束不同，不能把HAB的超力指标直接写为本任务原生指标。

## 10 · 向老师汇报的讲解稿

我们首先复现了ICLR 2024的TD-MPC2，在Meta-World Pick-Place V2任务上建立世界模型控制基线。模型从随机权重开始，不使用教师示范，输入是39维状态，通过世界模型预测候选动作后果，再由MPC选择动作执行。

本轮完成50万原生环境步训练，用时约4小时4分钟。正式最终评估10个完整回合，8个在最后一步满足成功条件，成功率80%；阶段最好是90%，两者分别报告。展示中同时保留成功补录和正式失败回合，补录不计入正式成绩。

目前完成的是一个任务、一个种子、缩减预算的原方法复现，尚未加入自己的算法创新。下一步将扩大重复实验，并检验能否用动作后果监督改善候选排序与最终任务完成。

## 11 · 证据与阅读入口

- [完整阶段结果与复现说明](tdmpc2_original/reports/50万步复现结果.md)
- [算法流程、训练评估与改动边界](tdmpc2_original/reports/算法流程与复现边界.md)
- [启动入口与预算记录代码](tdmpc2_original/runner.py)
- [原方法MPC与学习更新](tdmpc2_original/upstream/tdmpc2/tdmpc2.py)
- [官方在线训练器](tdmpc2_original/upstream/tdmpc2/trainer/online_trainer.py)
- [演示说明](tdmpc2_original/reports/演示说明.md)
- [已发布的原演示包](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.5.0-metaworld-original-demo)

正文、曲线和目录可离线阅读；视频需保留相邻媒体目录，换电脑请解压便携包后打开根目录HTML。按Ctrl+P可打印或另存为PDF，PDF不保留视频播放功能。论文及GitHub外链需要联网。
'''
md=md.replace('__TABLE__',table).replace('__CHART__',chart).replace('__VIDEOS__','\n\n'.join(videos))
(ROOT/'reports/MetaWorld_TD-MPC2复现与实验结果汇报.md').write_text(md,encoding='utf-8',newline='\n')
print(json.dumps({'stages':len(rows),'final_success':rows[-1]['episode_success'],'verified_demo_videos':len(videos)},ensure_ascii=True))
