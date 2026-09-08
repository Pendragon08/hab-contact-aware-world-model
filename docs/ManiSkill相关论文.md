# 使用 ManiSkill 开展仿真实验的论文：七篇核查与课题关联

核查日期：2026-09-08。范围：2024—2026 年正式发表、实际在 ManiSkill 系列环境运行训练或评测的精选论文。本次确认的七篇发表于 2024—2025 年，属于所要求的时间范围；这不是穷尽清单。纳入 5 篇 ICLR、1 篇 ICML、1 篇 CoRL 主会论文。会议论文不使用期刊影响因子或中科院分区标注。

七篇都包含实际仿真实验。必须区分 ManiSkill2、ManiSkill3/MS-HAB 和 SIMPLER 的 real2sim 分支；原论文使用的版本与今天仓库最新支持的版本可能不同。仅有引用、仅共享 SAPIEN 引擎或仅出现在官网成果页，均不足以证明使用了 ManiSkill 实验。

## 1. 总览

|论文|正式发表|ManiSkill 实验内容|对当前课题的用途|
|---|---|---|---|
|[ManiSkill-HAB](https://proceedings.iclr.cc/paper_files/paper/2025/hash/27aa3a0e6d63db269977bb2df5607cb8-Abstract-Conference.html)|ICLR 2025 主会|Fetch；TidyHouse、PrepareGroceries、SetTable；Pick、Place、Open、Close 子任务|当前Fetch环境与评测协议|
|[TD-MPC2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)|ICLR 2024 主会|Lift Cube、Pick Cube、Pick YCB、Stack Cube、Turn Faucet，共 5 个 ManiSkill2 任务|世界模型与潜空间规划基线|
|[RFCL](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html)|ICLR 2024 主会|PickCube、StackCube、PegInsertionSide；额外困难任务 PlugCharger|少示范、稀疏奖励下的训练效率|
|[DrS](https://proceedings.iclr.cc/paper_files/paper/2024/hash/22563ee9860728cac644b8cef32431cf-Abstract-Conference.html)|ICLR 2024 主会|Pick-and-Place：74个YCB训练物体、1600个EGAD测试物体；Turn Faucet：10个训练、50个测试水龙头；Open Cabinet Door：4扇训练、6扇测试柜门|跨物体复用的阶段奖励|
|[CoTPC](https://proceedings.mlr.press/v235/jia24c.html)|ICML 2024 主会|正式论文§5.3：Stack Cube、Peg Insertion、Turn Faucet、Push Chair、Pour；Pour仅支持视觉观测|从示范学习子技能与动作|
|[SIMPLER](https://proceedings.mlr.press/v270/li25c.html)|CoRL 2024 主会；PMLR 论文集于 2025 出版|Google Robot 和 WidowX；抓罐、抽屉操作、放置及堆叠等 8 个任务族|仿真评测能否反映实机表现|
|[AHA](https://proceedings.iclr.cc/paper_files/paper/2025/hash/70a06501001e1820fd1eb9ee821302d2-Abstract-Conference.html)|ICLR 2025 主会|抓物体、Push T、放球、叠方块、开抽屉等 5 个下游任务；主要训练数据来自 RLBench|失败识别与下游奖励反馈|

## 2. 每篇的五项结构化总结

以下“我们的分析”表示本次调研判断，区别于作者在正文明确陈述的局限。

### 1. ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks

**发表与版本：** ICLR 2025 主会；基于 ManiSkill3 的 MS-HAB；与当前 demo 直接相关。

**实际仿真任务：** Fetch；TidyHouse、PrepareGroceries、SetTable；Pick、Place、Open、Close 子任务。

- **解决的问题：** 原家庭整理基准的低层接触控制、吞吐和示范生成能力不足。
- **创新点：** 将 HAB 实现为 GPU 并行、真实夹持的低层操作基准，并用仿真事件自动分类和筛选轨迹。
- **采用的方法：** Pick/Place 使用 SAC，Open/Close 使用 PPO；深度图与状态融合，技能串接，并训练模仿学习基线。
- **评估标准与指标：** §5.1：每次正式评估 1000 回合；子任务统计 200 步内至少一次成功的 success once rate；长任务用 progressive completion rate。另报 SPS 和显存。Table 1 的 TidyHouse Pick 验证集 RL-ALL 为 68.15%，RL-PER 为 77.48%，不能与本项目的有限布局开发评估直接比较。
- **局限性：** 作者 §7 不声称迁移到真实机器人；原论文以理想任务规划器和瞬移导航聚焦操作（§3.2、§5.1）。我们的分析：不能据此声称解决完整自主移动导航；场景泛化、拥挤空间和技能交接仍有不足。

**核查位置：** PDF p1 基于 ManiSkill3；§3.1–3.3；§5.1；Table 1；§7。代码后续加入的导航策略不能倒算为原论文实验。

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/27aa3a0e6d63db269977bb2df5607cb8-Abstract-Conference.html) · [官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/27aa3a0e6d63db269977bb2df5607cb8-Paper-Conference.pdf) · [代码](https://github.com/arth-shukla/mshab) · [项目页](https://arth-shukla.github.io/mshab/)

### 2. TD-MPC2: Scalable, Robust World Models for Continuous Control

**发表与版本：** ICLR 2024 主会；ManiSkill2；是世界模型控制算法实验。

**实际仿真任务：** Lift Cube、Pick Cube、Pick YCB、Stack Cube、Turn Faucet，共 5 个 ManiSkill2 任务。

- **解决的问题：** 世界模型控制跨任务调参多，放大模型和多任务训练时不稳定。
- **创新点：** 整合 SimNorm、奖励/价值分布回归、Q 集成和最大熵策略先验，提高潜空间控制的扩展性。
- **采用的方法：** 学习编码器、动作条件潜动力学、奖励和价值头；以 MPPI 在潜空间优化动作序列，执行首步后重新规划。
- **评估标准与指标：** §4、Figure 14：ManiSkill2 用成功率—环境步数曲线，3 个种子均值及 95% 置信区间；Pick YCB 预算 14M 步，其余四任务 4M。全文另含其他仿真域，总计 104 项在线任务，不能说全部是 ManiSkill。
- **局限性：** 作者讨论奖励误设、安全和大规模数据成本。我们的分析：Q 分歧并非校准后的掉落概率；原文没有在 ManiSkill-HAB 的家庭长任务上验证该算法。

**核查位置：** 正式 PDF §4、Figure 1/4、p23 Figure 14 明确列出 5 个 ManiSkill2 任务。

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html) · [官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf) · [代码](https://github.com/nicklashansen/tdmpc2) · [项目页](https://www.tdmpc2.com/)

### 3. Reverse Forward Curriculum Learning for Extreme Sample and Demo Efficiency

**发表与版本：** ICLR 2024 主会；ManiSkill2；代码后续新增 ManiSkill3 支持，不代表原论文使用 ManiSkill3。

**实际仿真任务：** PickCube、StackCube、PegInsertionSide；额外困难任务 PlugCharger。

- **解决的问题：** 稀疏奖励中的精细操作难以探索；少量示范覆盖的初始状态有限，直接模仿难以泛化。
- **创新点：** 逐示范反向课程与正向课程串联，分别解决接近成功和扩大初始状态覆盖的问题，并动态设置回合时限。
- **采用的方法：** 先沿成功示范重置仿真状态，从接近终点逐步向前训练 SAC；再优先采样当前策略偶尔能成功的初始状态，扩大可解范围。
- **评估标准与指标：** 正文 §5、图3：5条示范、100万环境步、5随机种子，报告成功率及95%置信区间。图5为ManiSkill2的200万步示范量消融；附录B.1采用回合结束时的成功判据。21项主基准加额外PlugCharger才是22项总任务，并非全部属于ManiSkill。
- **局限性：** 作者 §6：依赖任意状态重置，通常限于仿真训练，实机部署仍需sim-to-real。我们的分析：主要证据来自状态输入任务，示范选择存在方差，不能直接推广为视觉输入或实机在线学习效果。

**核查位置：** §5明确使用ManiSkill2；附录B.1表4列出四个ManiSkill2任务及100步时限。正式会议题名使用Demo；arXiv较长的Demonstration Efficiency in Reinforcement Learning题名为同一论文。

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html) · [官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/cd062f8003e38f55dcb93df55b2683d6-Paper-Conference.pdf) · [代码](https://github.com/StoneT2000/rfcl) · [项目页](https://reverseforward-cl.github.io/)

### 4. DrS: Learning Reusable Dense Rewards for Multi-Stage Tasks

**发表与版本：** ICLR 2024 主会；正文称ManiSkill；官方实现明确使用ManiSkill2。

**实际仿真任务：** Pick-and-Place：74个YCB训练物体、1600个EGAD测试物体；Turn Faucet：10个训练、50个测试水龙头；Open Cabinet Door：4扇训练、6扇测试柜门。

- **解决的问题：** 手工稠密奖励需要领域知识和反复调参，模仿学习判别器收敛后又未必能复用为新实例的奖励。
- **创新点：** 以成功／失败监督替代专家／智能体分类，按任务阶段学习奖励，使奖励在策略收敛后仍保留进度信息。
- **采用的方法：** 通过二元阶段指示器划分轨迹，训练阶段判别器，并将有界输出加入阶段奖励；随后固定学到的奖励，在未见物体上训练SAC。
- **评估标准与指标：** §5.1按物体隔离训练与测试，图5比较成功率和样本效率；统一使用SAC，5随机种子，阴影为标准差。部分任务接近人工奖励，不能概括为全部超过人工奖励。
- **局限性：** 作者 §6：阶段结构如何自动获得尚未系统解决，阶段指示器也增加实机直接训练难度。我们的分析：复用主要为同一任务族内跨物体迁移，不等于跨任意任务和机器人使用的通用奖励。

**核查位置：** §5.1及图4列出三类ManiSkill任务与不重叠物体划分；官方README安装及实验命令使用mani_skill2包和drs_*_maniskill2.py。

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/22563ee9860728cac644b8cef32431cf-Abstract-Conference.html) · [官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/22563ee9860728cac644b8cef32431cf-Paper-Conference.pdf) · [代码](https://github.com/tongzhoumu/DrS) · [项目页](https://sites.google.com/view/iclr24drs)

### 5. Chain-of-Thought Predictive Control

**发表与版本：** ICML 2024 主会；ManiSkill2。

**实际仿真任务：** 正式论文§5.3：Stack Cube、Peg Insertion、Turn Faucet、Push Chair、Pour；Pour仅支持视觉观测。

- **解决的问题：** 接触密集操作的次优示范存在噪声、依赖历史的行为与动作不连续，普通行为克隆难以泛化。
- **创新点：** 无监督发现子技能边界，以边界观测构成子技能链，联合学习子技能特征和低层动作。
- **采用的方法：** 用PELT变点检测与动作余弦距离切分示范，在Behavior Transformer中加入可学习CoT标记和混合注意力掩码，联合优化模仿与子技能预测损失。
- **评估标准与指标：** §5.3、表2/3：报告状态与点云输入的任务成功率，区分已见物体的新场景与未见几何的零样本测试。主表选择3次训练、末20个检查点中的最好结果；附录L另报均值和标准差，不能把主表当成均值。
- **局限性：** 作者 §6：相似连续动作对应同一子技能的假设可能被精细微调动作破坏，离线学习仍受示范质量限制。我们的分析：这里的CoT是辅助训练的隐式技能特征，不是LLM语言推理链；推理时不执行显式CoT解码器。

**核查位置：** 2024年正式论文§5.3直接列出五个ManiSkill2任务。旧代码README含PickCube而无Pour，本报告以正式论文任务清单为准。

[正式论文](https://proceedings.mlr.press/v235/jia24c.html) · [官方 PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/jia24c/jia24c.pdf) · [代码](https://github.com/SeanJia/CoTPC) · [项目页](https://sites.google.com/view/cotpc)

### 6. Evaluating Real-World Robot Manipulation Policies in Simulation

**发表与版本：** CoRL 2024 主会；PMLR 论文集于 2025 出版；SIMPLER，基于 ManiSkill2 的 real2sim 分支。

**实际仿真任务：** Google Robot 和 WidowX；抓罐、抽屉操作、放置及堆叠等 8 个任务族。

- **解决的问题：** 实机策略评测昂贵，仿真分数又未必保持真实策略排名。
- **创新点：** 以系统辨识、视觉匹配/变体聚合降低评测差距，并提出 MMRV 排名错误指标。
- **采用的方法：** 构建对应真实机器人任务的仿真环境，运行已有策略，进行配对仿真—真实比较。
- **评估标准与指标：** 超过 1500 次配对 sim-real 评测；指标包括任务成功率、Pearson r、MMRV（越低越好）；前两者分别衡量执行与分数相关，MMRV 衡量错排造成的真实性能差距。
- **局限性：** 作者指出刚体、相机/视觉匹配及系统辨识范围的限制。我们的分析：有限机器人和策略集合上的一致性不能保证未知接触动力学风险或实机安全。

**核查位置：** 论文 §3–5 与附录 G；官方仓库 Introduction 明示基于 SAPIEN + CPU ManiSkill2，子模块 ManiSkill2_real2sim。后续 ManiSkill3 分支不是原论文版本。

[正式论文](https://proceedings.mlr.press/v270/li25c.html) · [官方 PDF](https://raw.githubusercontent.com/mlresearch/v270/main/assets/li25c/li25c.pdf) · [代码](https://github.com/simpler-env/SimplerEnv) · [项目页](https://simpler-env.github.io/)

### 7. AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation

**发表与版本：** ICLR 2025 主会；ManiSkill-Fail 离线评估 + ManiSkill 中的下游 RL 奖励改进实验。

**实际仿真任务：** 抓物体、Push T、放球、叠方块、开抽屉等 5 个下游任务；主要训练数据来自 RLBench。

- **解决的问题：** 通用视觉语言模型难检测机器人操作失败并解释原因。
- **创新点：** FailGen 自动合成失败，结合 AHA 指令微调，再用失败解释改善机器人系统反馈。
- **采用的方法：** 从成功示范注入动作/目标等故障，微调 LLaVA 类模型；将失败反馈用于 Eureka 奖励修改、任务规划和子任务验证。
- **评估标准与指标：** §5.1–5.3：ManiSkill-Fail 上比较成败判断准确率、ROUGE-L、余弦相似度和 LLM 语义判分；ManiSkill 下游实验另看真实执行的任务成功率，不能把检测准确率当机器人成功率。
- **局限性：** 作者说明故障覆盖与训练分类相关。我们的分析：语言解释相似度不等于因果正确性；合成失败仍有域差。公开仓库给出原规模微调约 8×A100 80GB、40 小时，不宜将完整复现默认安排到两张 A6000。

**核查位置：** 正式论文 §5.2、§5.3、附录 §7.6；作者网站展示五个 ManiSkill 下游任务。仓库 FailGen 主数据生成基于 RLBench，不能说全在 ManiSkill 训练。

[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/70a06501001e1820fd1eb9ee821302d2-Abstract-Conference.html) · [官方 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/70a06501001e1820fd1eb9ee821302d2-Paper-Conference.pdf) · [代码](https://github.com/NVlabs/AHA) · [项目页](https://aha-vlm.github.io/)

## 3. 与我们当前 ManiSkill-HAB demo 的关系

当前 demo 使用 Fetch 的 TidyHouse Pick 子任务及官方 SAC 策略，因此最应先读 ManiSkill-HAB，弄清任务成功条件、训练/验证场景划分、子任务与长任务的区别。原论文的长任务依赖理想任务规划器及瞬移导航；不能把它描述成已解决完整自主导航。参见 [ManiSkill-HAB 原论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/27aa3a0e6d63db269977bb2df5607cb8-Paper-Conference.pdf)。

如果继续聚焦世界模型，第二篇应读 TD-MPC2：它提供潜空间动力学与在线规划的算法依据，而且原论文确实在五项 ManiSkill2 任务上评测。它没有直接在 HAB 上验证，接到当前环境需要适配观测、动作、奖励、回合结束与重置语义。参见 [TD-MPC2 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf)。

接下来按研究问题选第三篇：训练样本不够读 RFCL；阶段奖励难设计读 DrS；希望分析子技能和历史依赖读 CoTPC；关注失败解释读 AHA；需要讨论仿真结论与实机的关系读 SIMPLER。这是针对当前项目的阅读建议，不代表已完成新颖性检索。

**可检验的课题切口（本次分析）：** 在保持 HAB 场景划分和动作/观测条件可比的前提下，研究世界模型的动作规划能否减少“抓取一度成功、随后又失败”的情况。先以官方 SAC 为基线，在相同预算下接入标准 TD-MPC2，再决定是否需要稳定持握约束、阶段奖励或失败恢复。仅把 TD-MPC2 接入 HAB 属于基线复现与迁移工作，论文贡献还需要明确的方法改进和受控实验支持。

建议同时记录官方 success once、回合结束成功率，以及人为预先定义的抓取后掉落率；后者应先明确“有效抓取”“失去抓持”的接触条件和时间阈值，并核验回放，不能直接把 once 与 end 的差值叫作掉落率。长任务再增加 progressive completion rate。比较时还应固定场景/物体分布、预算、评测种子，并报告多次独立训练的均值和不确定性。

本报告最初对应五回合 Demo；项目随后增加了 100 回合基线及新的接触控制开发实验。它们均不能直接当作 HAB Table 1 的完整正式协议结果。当前状态以仓库 README 和实验报告为准。

## 4. 使用文献时容易混淆的细节

- TD-MPC2 全文有 104 项在线任务，其中只有 5 项是 ManiSkill2；不能写成 104 项 ManiSkill 任务。
- RFCL 的原始论文使用 ManiSkill2；仓库后来提供 ManiSkill3 支持不改变原始实验版本。
- DrS 的跨物体奖励迁移并非任意任务之间的通用奖励迁移。
- CoTPC 的 CoT 指子技能链的隐式表示，不表示必须调用 LLM API。正式论文任务表与旧代码 README 不完全一致，复现前需核对对应配置。
- SIMPLER 属于以 ManiSkill2 为基础的 real2sim 实验环境。CoRL 会议年份是 2024，PMLR 论文集出版年份是 2025。
- AHA 的主要训练数据生成基于 RLBench，但另有 ManiSkill-Fail 评估及 ManiSkill 下游闭环强化学习实验。失败检测准确率不能写成机器人任务完成率。其官方仓库报告原规模微调约需 8×A100 80GB、40 小时，因此不建议把完整训练当作两张 A6000 的第一步。[官方训练说明](https://github.com/NVlabs/AHA)

## 5. 文件与记录

本仓库保留各论文的正式发表链接、官方 PDF 和开源代码链接。原始本地 PDF 仍保存在用户的文献目录，GitHub 版本使用上方官方链接访问，不依赖本机盘符。结构化核查记录见 [核查记录](ManiSkill_论文核查记录.json)。
