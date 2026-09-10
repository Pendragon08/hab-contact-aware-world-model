# TD-MPC2：算法流程、训练与测试

**论文：** TD-MPC2: Scalable, Robust World Models for Continuous Control  
**作者：** Nicklas Hansen、Hao Su、Xiaolong Wang  
**发表：** ICLR 2024 · 解读依据：arXiv v2及官方实现 · 整理日期：2026年9月10日

> 💡 汇报主线：机器人先在学到的模型中比较候选动作的后果，再进入环境执行；训练时利用真实反馈继续改进模型。世界模型负责预测，策略先验提出候选，MPC负责搜索和选择。

## 01 · 一页理解整个算法

TD-MPC2是一套基于模型的强化学习算法。它学习对控制有用的潜在表示，在这个表示空间中推演动作后果，无需把未来状态解码成图像或视频。

<div role="img" aria-label="控制闭环：环境观察、编码、候选动作、模型预测、规划评分、执行首动作、取得新观察后重复" style="padding:22px;background:#eef4fb;border:1px solid #cbd9ec;border-radius:12px"><div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-family:Arial,'Microsoft YaHei',sans-serif;font-size:16px;line-height:1.6"><div style="background:#fff;border:1px solid #cad7e8;border-radius:8px;padding:12px">① 环境观察<br><strong>读取真实反馈</strong></div><span aria-hidden="true">→</span><div style="background:#fff;border:1px solid #cad7e8;border-radius:8px;padding:12px">② 编码器<br><strong>形成潜在状态</strong></div><span aria-hidden="true">→</span><div style="background:#fff;border:1px solid #cad7e8;border-radius:8px;padding:12px">③ 候选序列<br><strong>采样＋策略先验</strong></div><span aria-hidden="true">→</span><div style="background:#dceafd;border:1px solid #a9c4e9;border-radius:8px;padding:12px">④ 世界模型<br><strong>预测状态与奖励</strong></div><span aria-hidden="true">→</span><div style="background:#dceafd;border:1px solid #a9c4e9;border-radius:8px;padding:12px">⑤ MPC搜索<br><strong>评分并迭代采样</strong></div><span aria-hidden="true">→</span><div style="background:#dff2e9;border:1px solid #a9d1be;border-radius:8px;padding:12px">⑥ 环境执行<br><strong>只执行首动作</strong></div></div><p style="margin:16px 0 0;color:#23466c"><strong>闭环反馈：</strong>取得新的环境观察 → 回到①重新规划。训练时，实际交互数据还会进入经验回放池，用于更新网络。</p></div>

**两种“试一试”必须区分：**规划阶段是在神经网络里计算候选后果；选定后才在仿真环境或真实机器人上执行。原版MPC不会为每个候选都重置仿真、实际运行一遍。

## 02 · 五个网络模块，分别做什么

| 模块 | 输入 → 输出 | 它回答的问题 |
|---|---|---|
| 编码器 h | 观察 s → 潜在表示 z | 怎样把环境信息变成网络使用的表示？ |
| 潜在动力学 d | z、动作 a → 下一潜在表示 z′ | 执行这个动作，接下来会怎样？ |
| 奖励模型 R | z、a → 预测即时奖励 | 这一步可能获得多少奖励？ |
| 价值网络 Q | z、a → 预测长期回报 | 从这里开始，后续可能获得多少折扣奖励？ |
| 策略先验 p | z → 候选动作分布 | 什么动作值得优先尝试？ |

论文将策略先验也列入整体架构，因此图中会出现“预测动作”。但动作由策略先验提出，下一状态由动力学模型预测，两者职责不同。**Q值是回报估计，不是成功概率。**

依据：[原论文§3.1与Figure 3](https://arxiv.org/html/2310.16828v2#S3.SS1)、[官方网络代码](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/tdmpc2/common/world_model.py)。

## 03 · 每一个控制步：如何规划动作

1. **编码当前观察。**用编码器获得当前潜在状态。
2. **生成候选动作序列。**从可调整的高斯分布中采样，并加入策略先验生成的序列；前一步规划结果可用于初始化本次搜索。
3. **预测候选后果。**动力学模型沿每个序列逐步推演，奖励模型预测各步奖励。
4. **估计总回报。**把短期预测奖励与规划窗口末端的价值估计结合。
5. **迭代改进采样。**根据高分候选及其权重更新采样分布，固定轮数地重复搜索。论文采用MPPI这种无需对动作求梯度的采样优化。
6. **执行首动作。**从最终搜索结果选出动作，只执行序列第一步；下一控制步重新观察和规划。

<div style="padding:18px 22px;background:#f8f5ec;border-left:4px solid #b8390e;border-radius:6px;font-size:18px;line-height:1.8"><strong>候选得分 ≈ 未来H步的折扣预测奖励 ＋ 第H步状态的折扣后续价值</strong><br><span style="font-size:15px;color:#555">概念式：G = Σ γᵏ R(zₖ,aₖ) + γᴴ Q(zᴴ,p(zᴴ))，其中k从0到H−1。实际实现还包含策略采样与Q集成等细节。</span></div>

这里的“末端”是**规划窗口的末端**，不是整项任务的结束。Q网络用来补充窗口之外的收益，所以短时域规划仍会考虑更远的未来；估计可能有误，并不保证找到最优动作。

依据：[原论文§3.2](https://arxiv.org/html/2310.16828v2#S3.SS2)、[固定版本官方规划代码：plan与_estimate_value](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/tdmpc2/tdmpc2.py)。

## 04 · 训练：模型如何从不会到会

单任务从头训练时，网络初始没有任务能力。先用随机动作收集初始经验，之后交替进行环境交互、经验保存与网络更新。**不要求先有教师模型，也不要求先完成独立预训练阶段。**论文另外研究了多任务离线训练与新任务适配，不能与单任务从头在线训练混为一谈。

每次交互记录：**当前观察s、实际动作a、真实奖励r、下一观察s′**。数据进入经验回放池，训练时抽取连续片段。

| 更新内容 | 学习目标 | 如何理解 |
|---|---|---|
| 动力学与表示 | 预测的下一潜在表示接近真实下一观察的编码；目标分支停止梯度 | 让模型对动作后果的预测更一致 |
| 奖励预测 | 预测奖励接近环境给出的真实奖励 | 学会估计眼前收益 |
| 价值预测 | 学习“真实奖励＋折扣后的下一步目标价值” | 从经验中逐步估计更长远的收益 |
| 策略先验 | 提高提出动作的Q估计，并保留动作多样性 | 更快地提出值得搜索的候选 |

编码器、动力学、奖励和价值模块通过联合目标学习；策略先验按自己的目标更新。目标Q网络采用缓慢移动平均更新，减轻学习目标的波动。策略先验**不是默认通过模仿教师或照抄MPC动作训练的行为克隆模型**。

训练闭环：**交互 → 保存经验 → 抽样更新 → 用更新后的模型继续规划 → 继续交互。**

依据：[官方在线训练器](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/tdmpc2/trainer/online_trainer.py)、[官方更新函数：update与update_pi](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/tdmpc2/tdmpc2.py)。

## 05 · 为什么称为Scalable、Robust

| 关键设计 | 主要作用 |
|---|---|
| SimNorm | 对潜在表示分组归一化，帮助稳定训练 |
| 奖励与价值的分布式回归 | 在变换后的数值区间上学习软分类目标，适应不同奖励尺度 |
| Q网络集成与缓慢更新的目标网络 | 减轻价值估计偏差和训练波动 |
| 多任务嵌入与动作维度掩码 | 区分任务并支持不同观察、动作空间 |

这里的Robust主要对应算法训练和跨任务表现的稳健性，**不等于接触安全保证**。本文的重点是算法流程；这些机制是支撑它稳定运行和扩展的设计。依据：[原论文§3.1、§3.3及附录A](https://arxiv.org/html/2310.16828v2)。

## 06 · 测试时还做哪些事情

| 环节 | 训练阶段 | 常规评估阶段 |
|---|---|---|
| 读取观察、编码 | 是 | 是 |
| 候选采样、模型推演、MPC搜索 | 初始随机收集之后使用 | 是 |
| 在环境执行动作 | 是 | 是 |
| 使用经验进行梯度更新 | 是 | 否 |
| 网络参数 | 随训练变化 | 固定 |
| 主要输出 | 模型权重、学习曲线、交互数据 | 回报、成功率、评估轨迹 |

**参数固定不等于动作固定。**测试时MPC仍根据每一步的新观察重新计算动作。规划优化的是候选动作分布，而不是重新训练神经网络权重。

评估要采用任务原生标准。HAB的“最终成功”不能仅用“曾抓住物体”替代；多任务比较还要区分各环境的回报和成功定义。依据：[官方训练器中的eval](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/tdmpc2/trainer/online_trainer.py)。

## 07 · 抓方块示例：预测、选择、执行

**以下是解释性示例，不是实验结果，也不表示潜在状态可以直接解码成这些物理描述。**

假设机械臂已经夹住方块，需要抬起并稳定保持：

1. 编码器整理当前机器人和物体的观察信息。
2. 策略先验提出向上抬的动作，规划器同时采样其他方向与幅度。
3. 世界模型计算不同动作序列的潜在后果与奖励预测。
4. 价值网络补充短期预测之后的收益，MPC据此进行搜索和选择。
5. 仿真器执行首动作，返回物体实际运动与真实奖励。
6. 训练时，用实际数据改进网络；测试时，只记录结果并继续规划。

物体会不会掉落，最终需要看环境中的实际执行。世界模型预测不是事实，也不是安全证明。

## 08 · 与我们的课题怎么衔接

| 层次 | 原论文已有内容 | 我们的工作定位 |
|---|---|---|
| 基础算法 | 潜在动力学、奖励、Q、策略先验与MPC | 建立原方法基线 |
| 仿真任务 | 包括Meta-World、ManiSkill2等 | 当前项目适配Meta-World与ManiSkill-HAB；HAB不是原论文同一基准 |
| 新的候选评价 | 原版按预测回报进行搜索评价 | 拟加入动作后果评分与相对排序监督，尚待实现验证 |
| 额外分支数据 | 原版规划使用神经网络推演 | 我们拟在数据采集阶段从同一仿真状态实际执行替代动作，获得结果标签 |

> 📝 新方案的研究问题：在相同候选与数据预算下，世界模型预测表示能否帮助选择更可能最终完成任务的动作？第一版冻结基础模型，主要改进候选评价与选择，不能直接声称改进了世界模型本体的训练方法。

**当前报告讲解原论文算法，不报告服务器实时进度，也不把拟议改进写成已完成成果。**

## 09 · 汇报时可直接使用的讲解稿

TD-MPC2通过环境交互学习一个服务于控制的潜在世界模型。机器人每一步先读取当前观察，再产生多组候选动作序列。世界模型预测这些动作的短期后果与奖励，价值网络补充更远的收益，MPC经过多轮采样比较后选出动作，只执行第一步，再根据新观察重新规划。

训练时，真实交互数据进入经验回放池，用于更新动力学、奖励、价值和策略先验；测试时网络参数固定，但规划持续进行。我们希望在这套框架上研究更准确的动作后果评价，让机器人在接触约束下更可靠地完成任务，而不仅是获得较高的预测回报。

## 10 · 论文、代码与阅读顺序

- [论文PDF：arXiv v2](https://arxiv.org/pdf/2310.16828v2)
- [论文网页版](https://arxiv.org/html/2310.16828v2)
- [作者项目：视频、数据与模型](https://www.tdmpc2.com/)
- [官方GitHub仓库](https://github.com/nicklashansen/tdmpc2)
- [本报告对应的固定代码版本](https://github.com/nicklashansen/tdmpc2/tree/57158282b46ebc5c329c5be9cfe2b0094126d1ca)

**推荐阅读顺序：Figure 3 → §3.1网络与训练 → §3.2规划 → §3.3多任务扩展 → 附录H实现细节。**

本报告为中文教学整理，并非作者原文译稿。报告正文和流程图可离线阅读；访问论文与代码链接需要联网。浏览器中按Ctrl+P可打印或另存为PDF。
