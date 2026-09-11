# Meta-World：TD-MPC2复现与实验结果

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

### 03.1 · 实际轨迹：把一次尝试记成连续的经历

这里的“实际”指**动作确实在 Meta-World 仿真环境中执行过**，不是现实机器人，也不是世界模型在内部想象出来的候选轨迹。一次动作交互可理解为一条记录：`（当前状态 s_t，实际动作 a_t，环境奖励 r_t，下一状态 s_{t+1}）`。

| 记录内容 | 在本次抓取放置任务中是什么 | 用来学什么 |
|---|---|---|
| 当前状态 s_t | 39维环境观测，包含机械臂、物体、目标等状态信息 | 知道从什么情况出发 |
| 实际动作 a_t | 4维连续控制：末端三方向移动与夹爪控制 | 知道机器人真正做了什么 |
| 环境奖励 r_t | 执行该动作时，由环境给出的奖励；本轮相加两次动作重复的奖励 | 监督奖励预测，并构造价值学习目标 |
| 下一状态 s_{t+1} | 动作重复执行2次后获得的39维观测 | 检查世界模型预测的动作后果是否准确 |

从重置环境开始，把100次动作决策按时间顺序连起来，就是一个完整回合：`s_0 → a_0、r_0 → s_1 → … → a_99、r_99 → s_100`。因此一个回合包含101个观测、100个动作和100个决策奖励，对应200个原生环境步。

成功和失败的训练回合都会保留，不是只学习成功视频。视频只是演示材料；训练读取数值状态、动作和奖励，不读取视频帧。环境的 success 用于记录和评估，这版更新没有把“成功/失败标签”作为单独的分类损失。

### 03.2 · 经验回放池：保存以前做过的尝试，反复拿来学习

可以把回放池理解为“练习记录库”。当前回合先临时保存；回合结束后，把完整轨迹加入回放池，并附上回合编号，用来区分不同尝试。随着训练进行，池中同时存在早期随机动作和后来模型规划动作产生的经验，已经学习过的经验也可以再次被抽到。

**完整回合入池，不等于每个回合只更新一次。** 比如第11个回合正在执行，模型仍可从前10个已完成回合中抽数据更新；第11回合的新数据等该回合结束入池后，才能被后续更新使用。回放池有容量上限，不是无限增长；也不等于永久保存的轨迹文件归档。

本轮实现将相邻观测与动作、奖励对齐存储：初始观测的动作和奖励是占位值；后续记录保存新观测及到达它所用的动作、所得奖励。采样时会移除片段首条记录对应的动作和奖励，使训练数据正确对应到“从当前状态执行动作，再到下一状态”。这是上述四元组的等价存储方式，不是少记了动作。

### 03.3 · 抽取片段：每次抽256段，每段包含连续3次动作

每次网络更新不会把所有历史回合从头学一遍，而是从回放池随机抽取一批连续片段。本轮 `batch_size=256`、`horizon=3`，所以一批包含256个片段；每个片段来自同一个回合，片段内部保持时间连续，不把不同回合的末尾与开头拼在一起。

一个片段的形状是：**4个观测＋3个动作＋3个奖励**，例如 `s_20 → a_20、r_20 → s_21 → a_21、r_21 → s_22 → a_22、r_22 → s_23`。

| 一批送入网络的数据 | 维度（时间 × 片段数 × 特征数） | 如何理解 |
|---|---|---|
| 观测 | 4 × 256 × 39 | 每段的起点，以及3次动作后的状态 |
| 动作 | 3 × 256 × 4 | 每段实际执行的3个动作 |
| 奖励 | 3 × 256 × 1 | 每个动作对应的环境奖励 |

256个片段不一定来自256个不同回合，也不保证互不重叠。一批包含768个动作转移的训练条目，**不表示此刻又在仿真环境执行了768步**；这些都是已收集的数据。保持连续是为了训练模型沿实际动作向前推演3步，并检查误差是否累积。片段长度3也不表示机器人完成抓取只需要3步。

### 03.4 · 更新网络：拿预测和学习目标对照，调整参数

先把片段起点的39维观测编码成512维潜在状态，再按片段里**当时实际执行的动作**在世界模型中推演3步。训练这段历史经验时，不需要让MPC重新决定该段动作。每个后续真实观测也会被编码，用作潜在状态预测的对照；这个对照分支停止梯度，避免两边同时追着彼此变化。

| 模块 | 如何利用片段学习 | 对后续控制有什么帮助 |
|---|---|---|
| 编码器＋状态转移模型 | 从片段起点连续预测未来潜在状态，与后续真实观测的编码比较，计算一致性误差 | 学习与动作后果有关的状态表示和动力学 |
| 奖励预测网络 | 预测每步奖励，与片段记录的环境奖励比较 | 为规划候选动作估计短期收益 |
| 5个Q价值网络 | 对齐“已获得的奖励＋折扣后的下一状态价值”这一TD目标 | 补充短期规划之后的长期收益估计 |
| 策略先验网络 | 在这些潜在状态上产生动作，利用Q评分和熵正则优化策略 | 为MPC提供更有希望的动作候选，并保留动作多样性 |
| 目标Q网络 | 由当前Q参数缓慢更新，作为价值目标的参考 | 降低价值学习目标快速变化造成的不稳定 |

Q的学习目标可以直观写成：`目标价值 = 实际奖励 + 0.95 × 目标Q对下一状态动作的价值估计`。其中下一状态动作由当前策略先验产生，价值项使用官方实现中的保守目标Q估计。它是逐步改善的自举目标，不是提前知道未来真实总回报，也不是成功概率。

程序先汇总带权的状态一致性、奖励和价值损失，通过反向传播及优化器更新编码器、动力学、奖励与Q网络；再单独更新策略先验，最后缓慢更新目标Q。奖励和Q使用官方的分布式数值编码及交叉熵损失，状态一致性使用均方误差。策略先验优化的是Q评分与熵目标，**没有直接模仿历史动作或教师动作的行为克隆损失**。

“更新参数”可以理解为根据预测错误微调网络内部的权重。优化器的一次学习更新不是一次环境动作，也不保证成功率立刻上升；仍需冻结模型、重新执行完整回合来评估效果。

### 03.5 · 抓取放置例子：一次失误怎样变成学习材料

下面是教学示意，不是从实验日志抽取的某个真实回合，也不代表这次复现采用了分阶段控制器。

| 环节 | 示意过程 |
|---|---|
| 在环境中尝试 | 机械臂靠近物体、闭合夹爪、尝试抬升；物体可能没有被带起来 |
| 记录实际后果 | 保存每一步的状态、控制指令、环境奖励和下一状态；不会只凭“发出了闭合指令”就认定抓取成功 |
| 完成回合并入池 | 整个回合完成后，连同失败经历一起存入回放池 |
| 抽到相关片段 | 某次更新恰好抽到闭合与抬升附近的连续3步 |
| 比较并学习 | 如果模型预测的潜在状态与实际观测编码不一致，就产生训练误差；奖励预测和Q估计也按各自目标修正 |
| 再次规划与检验 | 后续MPC使用更新后的模型给候选评分；是否真的更会抓取，要通过后续独立评估确认 |

仅凭某一个失败片段，模型不一定立即学会原因。片段也可能不覆盖整个抓取过程，需要多次交互、重复采样及价值传播逐步学习。

### 03.6 · 交互与学习的时间顺序：在线、离策略强化学习

<div style="padding:18px;background:#e9f2fb;border:1px solid #c7d9ed;border-radius:10px;line-height:2"><strong>随机探索积累初始经验 → 初始回放更新 → 循环：模型规划动作 → 仿真执行 → 当前回合临时记录 → 从已入池的历史回合抽256个片段 → 更新网络 → 继续交互。</strong><br>每当一个训练回合结束，完整轨迹加入回放池，供后续抽样。常规学习阶段基本每次动作决策对应一次学习更新；本轮一次决策重复执行2个原生环境步，不是每个原生步都更新一次。</div>

首次学习还有一次集中更新：本轮在决策索引1000处，使用已经入池的初始随机经验更新1000次；之后通常每次决策更新1次。因此，最初积累数据、首次集中学习和常规交替学习是三个不同阶段，不能把整个训练过程简单算成“每一步都更新一次”。

- **在线：**训练过程中仍不断与仿真环境交互、收集新数据；“在线”不是指联网或调用LLM API。
- **离策略：**当前网络可以利用早期随机控制或旧版本规划器产生的经验，而不要求样本全部由最新网络刚刚生成。
- **离线训练：**只使用预先固定的数据集训练，期间不再收集新的环境交互。本轮整体训练不属于这种情况；使用回放池不等于离线训练。
- **测试评估：**冻结网络，只执行动作并统计回报与成功率；不调用上述学习更新，也不把评估轨迹加入训练回放。

上述流程依据本轮固定版本的[在线训练器](tdmpc2_original/upstream/tdmpc2/trainer/online_trainer.py)、[回放池实现](tdmpc2_original/upstream/tdmpc2/common/buffer.py)和[网络更新实现](tdmpc2_original/upstream/tdmpc2/tdmpc2.py)。这是原方法的学习过程说明，没有新增训练模块或实验成绩。

### 03.7 · 本轮如何冻结评估

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
| 0 | 0/10 | 0% | 1.33 |
| 100,000 | 0/10 | 0% | 47.10 |
| 200,000 | 6/10 | 60% | 945.71 |
| 300,000 | 9/10 | 90% | 1,366.73 |
| 400,000 | 9/10 | 90% | 1,473.15 |
| 500,000 | 8/10 | 80% | 1,318.89 |

<figure style="margin:16px 0;padding:12px;background:white;border:1px solid #d4dce6;border-radius:10px"><svg xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" viewBox="0 0 792 273.6" xmlns="http://www.w3.org/2000/svg" version="1.1"> <metadata> <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"> <cc:Work> <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/> <dc:date>2026-09-10T16:06:49.113678</dc:date> <dc:format>image/svg+xml</dc:format> <dc:creator> <cc:Agent> <dc:title>Matplotlib v3.11.1, https://matplotlib.org/</dc:title> </cc:Agent> </dc:creator> </cc:Work> </rdf:RDF> </metadata> <defs> <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style> </defs> <g id="figure_1"> <g id="patch_1"> <path d="M 0 273.6 L 792 273.6 L 792 0 L 0 0 z " style="fill: #ffffff"/> </g> <g id="axes_1"> <g id="patch_2"> <path d="M 45.99649 237.598901 L 388.540625 237.598901 L 388.540625 19.030178 L 45.99649 19.030178 z " style="fill: #ffffff"/> </g> <g id="matplotlib.axis_1"> <g id="xtick_1"> <g id="line2d_1"> <path d="M 61.566678 237.598901 L 61.566678 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_2"> <defs> <path id="m1203d40bf0" d="M 0 0 L 0 3.5 " style="stroke: #000000; stroke-width: 0.8"/> </defs> <g> <use xlink:href="#m1203d40bf0" x="61.566678" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_1"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="61.566678" y="252.956322" transform="rotate(-0 61.566678 252.956322)">0</text> </g> </g> <g id="xtick_2"> <g id="line2d_3"> <path d="M 123.84743 237.598901 L 123.84743 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_4"> <g> <use xlink:href="#m1203d40bf0" x="123.84743" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_2"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="123.84743" y="252.956322" transform="rotate(-0 123.84743 252.956322)">100</text> </g> </g> <g id="xtick_3"> <g id="line2d_5"> <path d="M 186.128182 237.598901 L 186.128182 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_6"> <g> <use xlink:href="#m1203d40bf0" x="186.128182" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_3"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="186.128182" y="252.956322" transform="rotate(-0 186.128182 252.956322)">200</text> </g> </g> <g id="xtick_4"> <g id="line2d_7"> <path d="M 248.408933 237.598901 L 248.408933 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_8"> <g> <use xlink:href="#m1203d40bf0" x="248.408933" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_4"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="248.408933" y="252.956322" transform="rotate(-0 248.408933 252.956322)">300</text> </g> </g> <g id="xtick_5"> <g id="line2d_9"> <path d="M 310.689685 237.598901 L 310.689685 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_10"> <g> <use xlink:href="#m1203d40bf0" x="310.689685" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_5"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="310.689685" y="252.956322" transform="rotate(-0 310.689685 252.956322)">400</text> </g> </g> <g id="xtick_6"> <g id="line2d_11"> <path d="M 372.970437 237.598901 L 372.970437 19.030178 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_12"> <g> <use xlink:href="#m1203d40bf0" x="372.970437" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_6"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="372.970437" y="252.956322" transform="rotate(-0 372.970437 252.956322)">500</text> </g> </g> <g id="text_7"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="217.268557" y="267.957182" transform="rotate(-0 217.268557 267.957182)">Training environment steps (thousands)</text> </g> </g> <g id="matplotlib.axis_2"> <g id="ytick_1"> <g id="line2d_13"> <path d="M 45.99649 227.663959 L 388.540625 227.663959 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_14"> <defs> <path id="m2dbeefddd0" d="M 0 0 L -3.5 0 " style="stroke: #000000; stroke-width: 0.8"/> </defs> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="227.663959" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_8"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="231.84267" transform="rotate(-0 38.99649 231.84267)">0</text> </g> </g> <g id="ytick_2"> <g id="line2d_15"> <path d="M 45.99649 187.924191 L 388.540625 187.924191 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_16"> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="187.924191" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_9"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="192.102902" transform="rotate(-0 38.99649 192.102902)">20</text> </g> </g> <g id="ytick_3"> <g id="line2d_17"> <path d="M 45.99649 148.184423 L 388.540625 148.184423 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_18"> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="148.184423" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_10"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="152.363134" transform="rotate(-0 38.99649 152.363134)">40</text> </g> </g> <g id="ytick_4"> <g id="line2d_19"> <path d="M 45.99649 108.444655 L 388.540625 108.444655 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_20"> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="108.444655" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_11"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="112.623366" transform="rotate(-0 38.99649 112.623366)">60</text> </g> </g> <g id="ytick_5"> <g id="line2d_21"> <path d="M 45.99649 68.704887 L 388.540625 68.704887 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_22"> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="68.704887" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_12"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="72.883598" transform="rotate(-0 38.99649 72.883598)">80</text> </g> </g> <g id="ytick_6"> <g id="line2d_23"> <path d="M 45.99649 28.965119 L 388.540625 28.965119 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_24"> <g> <use xlink:href="#m2dbeefddd0" x="45.99649" y="28.965119" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_13"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="38.99649" y="33.14383" transform="rotate(-0 38.99649 33.14383)">100</text> </g> </g> <g id="text_14"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="11.357662" y="128.314539" transform="rotate(-90 11.357662 128.314539)">Success rate (%)</text> </g> </g> <g id="line2d_25"> <path d="M 61.566678 227.663959 L 123.84743 227.663959 L 186.128182 108.444655 L 248.408933 48.835003 L 310.689685 48.835003 L 372.970437 68.704887 " clip-path="url(#p43e1debd45)" style="fill: none; stroke: #136c60; stroke-width: 2.4; stroke-linecap: square"/> <defs> <path id="m6bf0980bd3" d="M 0 3 C 0.795609 3 1.55874 2.683901 2.12132 2.12132 C 2.683901 1.55874 3 0.795609 3 0 C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 C 1.55874 -2.683901 0.795609 -3 0 -3 C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 C -2.683901 -1.55874 -3 -0.795609 -3 0 C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 C -1.55874 2.683901 -0.795609 3 0 3 z " style="stroke: #136c60"/> </defs> <g clip-path="url(#p43e1debd45)"> <use xlink:href="#m6bf0980bd3" x="61.566678" y="227.663959" style="fill: #136c60; stroke: #136c60"/> <use xlink:href="#m6bf0980bd3" x="123.84743" y="227.663959" style="fill: #136c60; stroke: #136c60"/> <use xlink:href="#m6bf0980bd3" x="186.128182" y="108.444655" style="fill: #136c60; stroke: #136c60"/> <use xlink:href="#m6bf0980bd3" x="248.408933" y="48.835003" style="fill: #136c60; stroke: #136c60"/> <use xlink:href="#m6bf0980bd3" x="310.689685" y="48.835003" style="fill: #136c60; stroke: #136c60"/> <use xlink:href="#m6bf0980bd3" x="372.970437" y="68.704887" style="fill: #136c60; stroke: #136c60"/> </g> </g> <g id="patch_3"> <path d="M 45.99649 237.598901 L 45.99649 19.030178 " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/> </g> <g id="patch_4"> <path d="M 45.99649 237.598901 L 388.540625 237.598901 " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/> </g> <g id="text_15"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="61.566678" y="218.663959" transform="rotate(-0 61.566678 218.663959)">0/10</text> </g> <g id="text_16"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="123.84743" y="218.663959" transform="rotate(-0 123.84743 218.663959)">0/10</text> </g> <g id="text_17"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="186.128182" y="99.444655" transform="rotate(-0 186.128182 99.444655)">6/10</text> </g> <g id="text_18"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="248.408933" y="39.835003" transform="rotate(-0 248.408933 39.835003)">9/10</text> </g> <g id="text_19"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="310.689685" y="39.835003" transform="rotate(-0 310.689685 39.835003)">9/10</text> </g> <g id="text_20"> <text style="font-size: 10px; font-family: 'DejaVu Sans'; text-anchor: middle" x="372.970437" y="59.704887" transform="rotate(-0 372.970437 59.704887)">8/10</text> </g> <g id="text_21"> <text style="font-size: 13.2px; font-family: 'DejaVu Sans'; text-anchor: middle" x="217.268557" y="13.030178" transform="rotate(-0 217.268557 13.030178)">Terminal success</text> </g> </g> <g id="axes_2"> <g id="patch_5"> <path d="M 446.455625 237.598901 L 788.99976 237.598901 L 788.99976 19.030178 L 446.455625 19.030178 z " style="fill: #ffffff"/> </g> <g id="matplotlib.axis_3"> <g id="xtick_7"> <g id="line2d_26"> <path d="M 462.025813 237.598901 L 462.025813 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_27"> <g> <use xlink:href="#m1203d40bf0" x="462.025813" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_22"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="462.025813" y="252.956322" transform="rotate(-0 462.025813 252.956322)">0</text> </g> </g> <g id="xtick_8"> <g id="line2d_28"> <path d="M 524.306565 237.598901 L 524.306565 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_29"> <g> <use xlink:href="#m1203d40bf0" x="524.306565" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_23"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="524.306565" y="252.956322" transform="rotate(-0 524.306565 252.956322)">100</text> </g> </g> <g id="xtick_9"> <g id="line2d_30"> <path d="M 586.587317 237.598901 L 586.587317 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_31"> <g> <use xlink:href="#m1203d40bf0" x="586.587317" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_24"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="586.587317" y="252.956322" transform="rotate(-0 586.587317 252.956322)">200</text> </g> </g> <g id="xtick_10"> <g id="line2d_32"> <path d="M 648.868068 237.598901 L 648.868068 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_33"> <g> <use xlink:href="#m1203d40bf0" x="648.868068" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_25"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="648.868068" y="252.956322" transform="rotate(-0 648.868068 252.956322)">300</text> </g> </g> <g id="xtick_11"> <g id="line2d_34"> <path d="M 711.14882 237.598901 L 711.14882 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_35"> <g> <use xlink:href="#m1203d40bf0" x="711.14882" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_26"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="711.14882" y="252.956322" transform="rotate(-0 711.14882 252.956322)">400</text> </g> </g> <g id="xtick_12"> <g id="line2d_36"> <path d="M 773.429572 237.598901 L 773.429572 19.030178 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_37"> <g> <use xlink:href="#m1203d40bf0" x="773.429572" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_27"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="773.429572" y="252.956322" transform="rotate(-0 773.429572 252.956322)">500</text> </g> </g> <g id="text_28"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="617.727692" y="267.957182" transform="rotate(-0 617.727692 267.957182)">Training environment steps (thousands)</text> </g> </g> <g id="matplotlib.axis_4"> <g id="ytick_7"> <g id="line2d_38"> <path d="M 446.455625 237.598901 L 788.99976 237.598901 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_39"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="237.598901" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_29"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="241.777612" transform="rotate(-0 439.455625 241.777612)">0</text> </g> </g> <g id="ytick_8"> <g id="line2d_40"> <path d="M 446.455625 211.884933 L 788.99976 211.884933 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_41"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="211.884933" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_30"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="216.063644" transform="rotate(-0 439.455625 216.063644)">200</text> </g> </g> <g id="ytick_9"> <g id="line2d_42"> <path d="M 446.455625 186.170966 L 788.99976 186.170966 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_43"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="186.170966" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_31"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="190.349677" transform="rotate(-0 439.455625 190.349677)">400</text> </g> </g> <g id="ytick_10"> <g id="line2d_44"> <path d="M 446.455625 160.456998 L 788.99976 160.456998 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_45"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="160.456998" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_32"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="164.635709" transform="rotate(-0 439.455625 164.635709)">600</text> </g> </g> <g id="ytick_11"> <g id="line2d_46"> <path d="M 446.455625 134.743031 L 788.99976 134.743031 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_47"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="134.743031" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_33"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="138.921742" transform="rotate(-0 439.455625 138.921742)">800</text> </g> </g> <g id="ytick_12"> <g id="line2d_48"> <path d="M 446.455625 109.029063 L 788.99976 109.029063 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_49"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="109.029063" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_34"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="113.207774" transform="rotate(-0 439.455625 113.207774)">1000</text> </g> </g> <g id="ytick_13"> <g id="line2d_50"> <path d="M 446.455625 83.315096 L 788.99976 83.315096 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_51"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="83.315096" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_35"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="87.493807" transform="rotate(-0 439.455625 87.493807)">1200</text> </g> </g> <g id="ytick_14"> <g id="line2d_52"> <path d="M 446.455625 57.601129 L 788.99976 57.601129 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_53"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="57.601129" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_36"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="61.77984" transform="rotate(-0 439.455625 61.77984)">1400</text> </g> </g> <g id="ytick_15"> <g id="line2d_54"> <path d="M 446.455625 31.887161 L 788.99976 31.887161 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #b0b0b0; stroke-opacity: 0.18; stroke-width: 0.8; stroke-linecap: square"/> </g> <g id="line2d_55"> <g> <use xlink:href="#m2dbeefddd0" x="446.455625" y="31.887161" style="stroke: #000000; stroke-width: 0.8"/> </g> </g> <g id="text_37"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: end" x="439.455625" y="36.065872" transform="rotate(-0 439.455625 36.065872)">1600</text> </g> </g> <g id="text_38"> <text style="font-size: 11px; font-family: 'DejaVu Sans'; text-anchor: middle" x="404.818047" y="128.314539" transform="rotate(-90 404.818047 128.314539)">Return</text> </g> </g> <g id="line2d_56"> <path d="M 462.025813 237.427589 L 524.306565 231.543151 L 586.587317 116.009015 L 648.868068 61.879136 L 711.14882 48.19676 L 773.429572 68.028877 " clip-path="url(#p7b38345972)" style="fill: none; stroke: #205b9f; stroke-width: 2.4; stroke-linecap: square"/> <defs> <path id="m048a45494c" d="M 0 3 C 0.795609 3 1.55874 2.683901 2.12132 2.12132 C 2.683901 1.55874 3 0.795609 3 0 C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132 C 1.55874 -2.683901 0.795609 -3 0 -3 C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132 C -2.683901 -1.55874 -3 -0.795609 -3 0 C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132 C -1.55874 2.683901 -0.795609 3 0 3 z " style="stroke: #205b9f"/> </defs> <g clip-path="url(#p7b38345972)"> <use xlink:href="#m048a45494c" x="462.025813" y="237.427589" style="fill: #205b9f; stroke: #205b9f"/> <use xlink:href="#m048a45494c" x="524.306565" y="231.543151" style="fill: #205b9f; stroke: #205b9f"/> <use xlink:href="#m048a45494c" x="586.587317" y="116.009015" style="fill: #205b9f; stroke: #205b9f"/> <use xlink:href="#m048a45494c" x="648.868068" y="61.879136" style="fill: #205b9f; stroke: #205b9f"/> <use xlink:href="#m048a45494c" x="711.14882" y="48.19676" style="fill: #205b9f; stroke: #205b9f"/> <use xlink:href="#m048a45494c" x="773.429572" y="68.028877" style="fill: #205b9f; stroke: #205b9f"/> </g> </g> <g id="patch_6"> <path d="M 446.455625 237.598901 L 446.455625 19.030178 " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/> </g> <g id="patch_7"> <path d="M 446.455625 237.598901 L 788.99976 237.598901 " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/> </g> <g id="text_39"> <text style="font-size: 13.2px; font-family: 'DejaVu Sans'; text-anchor: middle" x="617.727692" y="13.030178" transform="rotate(-0 617.727692 13.030178)">Mean episode return</text> </g> </g> </g> <defs> <clipPath id="p43e1debd45"> <rect x="45.99649" y="19.030178" width="342.544135" height="218.568723"/> </clipPath> <clipPath id="p7b38345972"> <rect x="446.455625" y="19.030178" width="342.544135" height="218.568723"/> </clipPath> </defs> </svg> <figcaption>左：末步成功率；右：平均回合奖励。每点10回合，单个训练种子。连线仅连接阶段观测，不表示中间步数经过评估；未绘制跨种子置信区间。</figcaption></figure>

数据源为保存的metrics数值记录，图表由脚本直接生成。原始表可下载[阶段CSV](reports/metaworld_presentation/stage_results.csv)，核验记录见[结果JSON](reports/metaworld_presentation/verified_results.json)，曲线可单独导出[SVG](reports/metaworld_presentation/learning_curves.svg)或[PNG](reports/metaworld_presentation/learning_curves.png)。

## 05 · 这些结果说明什么

1. **已学到可执行的控制行为。**正式评估从初始0/10增加到最终8/10，配套保存了实际执行视频。成功不是仅由预测奖励推断出来的。
2. **最终结果是80%，不能写成90%。**90%是30万与40万步的阶段最好值；50万步检查点为80%。
3. **表现并非单调改善。**最后一轮比最好阶段少成功1个回合，每轮只有10回合，不能据此断言出现过拟合、灾难性遗忘或训练退化。
4. **这是可用的初步基线证据。**只有一个训练种子、一个任务；尚不能说明稳定达到原论文水平、具有跨任务泛化能力或优于其他算法。

这里不绘制原论文数字的“提升百分比”：预算、种子数量与评估协议没有完全对齐，不能强行横向排名。也不将旧Meta-World V3示范辅助实验和本轮V2成绩直接相减作为方法收益。

## 06 · 成功演示与失败案例

正式评估的最终10回合仍是8成功、2失败。原程序每轮只自动录制首回合，最终轮首回合恰好失败，因此后来使用**冻结最终权重**，按预先固定的101、102、103三个种子补录，三次均成功，优化器更新为0。补录不加入正式8/10，也不是新的大样本评估。

<figure style="margin:18px 0;padding:16px;background:#fff;border:1px solid #d4dce6;border-radius:10px"><figcaption><strong>补录成功 · 最终模型 · 种子101</strong> — 冻结最终权重的补充演示；不计入正式8/10。</figcaption><video controls playsinline preload="metadata" style="width:100%;max-height:410px;display:block;margin-top:12px;background:#14222b;border-radius:8px" src="tdmpc2_original/artifacts/demo/final_demo_seed101_103/seed_101.mp4">浏览器不支持视频，请通过演示包播放。</video></figure>

<figure style="margin:18px 0;padding:16px;background:#fff;border:1px solid #d4dce6;border-radius:10px"><figcaption><strong>正式评估失败 · 最终模型</strong> — 最终正式评估的首回合，属于2个失败回合之一。</figcaption><video controls playsinline preload="metadata" style="width:100%;max-height:410px;display:block;margin-top:12px;background:#14222b;border-radius:8px" src="tdmpc2_original/artifacts/demo/eval_250000.mp4">浏览器不支持视频，请通过演示包播放。</video></figure>

<figure style="margin:18px 0;padding:16px;background:#fff;border:1px solid #d4dce6;border-radius:10px"><figcaption><strong>训练前 · 随机初始化</strong> — 训练前评估首回合，用于展示尚未学会任务时的行为。</figcaption><video controls playsinline preload="metadata" style="width:100%;max-height:410px;display:block;margin-top:12px;background:#14222b;border-radius:8px" src="tdmpc2_original/artifacts/demo/eval_000000.mp4">浏览器不支持视频，请通过演示包播放。</video></figure>

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
