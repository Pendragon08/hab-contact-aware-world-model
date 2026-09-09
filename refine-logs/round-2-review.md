# 第 2 轮方法复审：首动作后果排序修订案

## 总体判断

修订稿已经解决第 1 轮最严重的方法错位：真实标签现在只干预 MPC 实际执行的首动作，之后由同一个冻结 π0 续跑；评分器也只使用当前 latent、首动作和一步预测 latent，不再依赖不会执行的候选尾部。方案明确承认标签依赖 π0、闭环 πθ 会改变分布，删除了阈值式安全保证，并把 HAB 累计量限制在 benchmark 指标含义内。这些修改使方案从“目标未对齐的风险头”变成了可以实现和证伪的动作结果排序研究。

Problem Anchor **已保留**；dominant contribution **明显更锋利**；方法 **显著更简单**；frontier leverage **恰当且没有强行堆叠新模型**。不过，候选池公平性、输出标签的逻辑结构、三基础种子的完整成本和最终新颖性仍未完全闭合，因此还不能达到 READY。

**CALIBRATION: none**

没有 curated known-good / known-bad 方案用于量表锚定。本复审仍是同系列模型的方案评议，不是独立文献裁决，也不是科学有效性证明。

## 评分

| 维度 | 分数（1–10） | 权重 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 9.0 | 15% | 1.35 |
| Method Specificity | 8.0 | 25% | 2.00 |
| Contribution Quality | 6.0 | 25% | 1.50 |
| Frontier Leverage | 8.0 | 15% | 1.20 |
| Feasibility | 6.0 | 10% | 0.60 |
| Validation Focus | 7.5 | 5% | 0.375 |
| Venue Readiness | 6.0 | 5% | 0.30 |
| **OVERALL SCORE** | **7.33 / 10** | **100%** | **7.325** |

## GAP

距离 READY 的 9 分门槛还有约 1.7 分。实现定义和验证逻辑已基本成形，剩余差距主要来自三点：当前双 BCE 头没有利用固定 Pick 任务中“有效成功、曾超限、其他失败”三类互斥结构；原版随机精英选择、同池 G-argmax 和新评分器仍未被写成完全相同 admissible first-action pool 上的比较；三基础种子需要各自采分支数据，完整出版级预算超过当前表格。更根本的 venue gap 取决于结果：只有冻结的一步 world-model feature 在同分支、同容量的 state-action critic 和 return critic 之上产生稳定增量，world-model 方法贡献才成立。

## 已解决的关键问题

1. **训练与部署对齐**：已从 3 步动作序列标签改成首动作干预，随后 π0 continuation；输入删除未执行尾部。
2. **policy-dependent 标签边界**：明确这是给定 π0 的结果标签，离线排序不能代替 πθ 闭环结果。
3. **安全与物理含义**：删除风险阈值和保证，承认累计力是固定控制频率下的 benchmark 累计量，不是冲量或损伤指标。
4. **分支回放**：覆盖物理状态、任务进度、随机数、控制器、MPC 缓存与延迟队列，并明确不能额外调用有累计副作用的 `evaluate()`。
5. **信息公平性**：学习型对照共享 branch labels、时间/累计量上下文和调参预算；无世界模型 critic、回报 critic、去配对和打乱 z' 均已进入验证。
6. **先导停止门槛**：E0 给出了异结果锚点和替代动作改善门槛，能在标签无支持度时及时停止。

## 仍需修订的低分维度

### Contribution Quality — 6.0 / 10

- **具体弱点**：相对 ContactGuard，首动作干预和重规划是自然延伸；相对普通 supervised critic，当前网络本身没有结构性新意。ContactGuard 不提供恢复/完成、FAWAM 研究力预测与残差反馈、ContactWorld 研究视触模型与规划，确实给本方案留下了“完成感知、固定候选、共同 continuation 的配对排序”这一窄缝，但实现他人 future work 或换成 HAB 指标并不自动形成顶会贡献。
- **具体修复**：把论文主张固定为一项：**同锚点首动作干预监督能否纠正 receding-horizon TD-MPC2 的候选结果排序。** 一步 latent、累计量条件和分类头都作为机制变量。预注册降级规则：若去 z' 或匹配信息的 state-action critic 持平，就停止“world-model feature 是必要机制”的主张，把成果定位为 paired action supervision / mechanism study。
- **优先级**：**CRITICAL**。

- **具体弱点**：`pS` 预测五条件成功，五条件已包含累计力未超限；`pV` 又预测曾超限。两个独立 BCE 头和 `pS-lambda*pV` 在固定 Pick 协议中重复编码同一不可逆结果。
- **具体修复**：在当前 `PickSubtaskTrain-v0` 中，task plan 仅含一个 Pick subtask；累计量只会在进入后续 subtask 时重置，而这里不存在后续 subtask。一旦累计量达到阈值，之后不可能满足含该条件的最终成功。因此可用一个三类 softmax `{最终有效成功, 曾累计力超限, 其他失败}`，再以 `U=p_success-lambda*p_force` 做同锚点 pairwise 排序。这样三类概率归一、避免不一致组合，并把共享 MLP 保持为唯一新组件。该互斥结论只适用于当前单 Pick 协议，不能直接外推到会在 subtask 间重置累计量的多技能 SequentialTask。
- **优先级**：**IMPORTANT**。

### Feasibility — 6.0 / 10

- **具体弱点**：E1 的约 32 万分支步只对应一个冻结基础 checkpoint。不同 TD-MPC2 训练种子的 latent 坐标和 π0 均不同，三基础种子需要各自采集约 32 万分支步，不能默认共享同一数据头。E2 当前按 3 方法计算 54 万步，但最终公平表至少需要原版、同池 G-argmax、最强匹配信息 critic、本文 4 方法，即 72 万步。
- **具体修复**：把预算分成两层。可执行先导为一个 checkpoint 的 E0/E1；出版级扩展明确计入约 `3×32万=96万` 分支步、`4×3×3×100×200=72万` 闭环评估步、两个新 50 万步基础种子和所有拟合。按当前仅供估算的 5.8 步/秒，分支约 46 小时、闭环约 35 小时、基础种子约 48 GPUh，加拟合和重配置后总量超过 130 GPUh。只有前两阶段通过才启动后者。
- **优先级**：**CRITICAL**。

- **具体弱点**：lambda 与 beta 的 12 组联合选择若只用 40 个模型选择锚点，且真正可比较 pair 很少，容易对小开发集过拟合；另留 40 锚点“经验校准”与首版不使用校准决策的简化目标不一致。
- **具体修复**：首版改为 240 train / 80 model-selection / 80 locked-test；Brier 可直接作为锁定测试上的描述性指标，无需拟合校准器。若未来真的做 calibration，再新增独立 calibration split 和预先固定的方法，不在 v1 同时承担。
- **优先级**：**IMPORTANT**。

### Venue Readiness — 6.0 / 10

- **具体弱点**：方案已经 paper-shaped，但尚未证明算法增量。一个 benchmark-specific 的单任务约束加小 critic，若只胜过原版而不胜过同数据无世界模型 critic，顶会审稿人仍会归因于额外特权标签或去随机化。
- **具体修复**：最终主表必须首先回答三个问题，而不是扩大任务列表：同池 G-argmax 是否已解决问题；同分支 state-action / return critic 是否已获得全部收益；加入冻结 z' 后是否还有跨基础种子、未见布局和执行偏移下的增量。只在第三问为肯定时升级为 world-model 方法论文。
- **优先级**：**CRITICAL**。

## 对高分维度的保留意见

### Method Specificity — 8.0 / 10

训练接口、标签来源、pairwise loss、分支恢复、调参集合、E0 门槛和闭环指标都已足够让工程实现启动。仍需把共同候选池写死：每个锚点的 4 条真实分支至少包含原版实际抽中的首动作、同批候选中 G 最大的首动作，以及两个预注册分层候选；首动作去重后按固定替补规则补齐。离线比较时，原版抽样、G-argmax、匹配 critic 和本文都必须从同一个去重后的 admissible first-action pool 选择。否则“固定 512”仍不足以排除选择规则混杂。

### Validation Focus — 7.5 / 10

三个实验块的功能已经清楚，且 E0 能保护预算。建议最终闭环保留四个方法：未经修改原版、同候选 G-argmax、E1 最强匹配信息 critic、本文。附录 J、规则项和所有消融停留在 E1 即可，避免 E2 膨胀。E0 的 8/40 与 4/40 是资源门槛，不应被写成统计显著性或方法通过证明；稿中已经正确区分。

## Simplification Opportunities

1. 用一个三类 outcome head 替换两个独立 BCE 头；保留一个共享 MLP、一个 CE 和一个同锚点 pairwise 项。
2. 删除首版独立 calibration split，把 40+40 合并为 80 个 model-selection 锚点；只报告自然分布测试上的描述性 Brier。
3. 将工作拆成“一个 checkpoint 的可执行机制先导”和“门槛通过后的后的三种子出版级验证”，避免初学者一开始承担超过 130 GPUh 的完整计划。

## Modernization Opportunities

**NONE。** 当前设计已正确复用冻结世界模型与规划缓存。再增加 VLM、Diffusion、力历史适应或生成式世界模型会破坏可归因性，也与计算边界不符。

## Drift Warning

**NONE。** Problem Anchor 完整保留。需要继续限制术语：结果只支持当前仿真、当前控制频率与原生累计量规则下的接触约束动作排序，不能外推为物理安全或真实损伤控制。

## 剩余具体行动项

1. 把双 BCE 输出改成仅适用于单 Pick 的三类互斥 outcome，并在文中写明为何不适用于多 subtask 累计量重置。
2. 写死同一 admissible first-action pool、原版实际抽样动作、G-argmax、分层候选和去重替补规则。
3. 将 E1 切分简化为 240/80/80，删除首版校准器与校准参与决策的任何暗示。
4. 将每基础种子独立分支采集与四方法闭环评估计入完整预算，并把 >130 GPUh 的出版级阶段放在 E0/E1 门槛之后。
5. 在最终 claim ladder 中保留明确降级规则：无世界模型 critic 持平时，不宣称 world-model feature 创新。
6. 实现前仍须通过 replay acceptance test；该测试通过只证明标签管线可信，不证明方法有效。

## Verdict

**REVISE**

修订稿已经从 5.63 提升到 7.33，方法方向和工程顺序可以继续。它尚未 READY 的原因是公平候选池、互斥标签建模和完整预算还需一次定稿，且 top-venue 贡献必须由匹配信息对照下的真实增量决定。无需 RETHINK，也不应增加新架构；完成上述五项收尾后即可进入有停止门槛的 E0 实施准备，但不能把“可实施”写成“论文贡献已验证”。
