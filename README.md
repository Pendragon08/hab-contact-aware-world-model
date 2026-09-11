# 世界模型与机器人抓取：TD-MPC2 复现及 ManiSkill-HAB 研究

研究世界模型如何帮助机器人规划动作，并在接触约束下完成抓取。当前包含 **Meta-World 原方法复现**、**ManiSkill-HAB 迁移基线**及历史对照实验。HAB 使用 Fetch 移动机械臂，当前覆盖抓取与回到休止姿态；完整仓储导航、搬运和放置尚未验证。

**建议先看：** [实验结果汇报](reports/MetaWorld_TD-MPC2复现与实验结果汇报.md) · [算法讲解](reports/TD-MPC2算法流程汇报.md) · [实验与报告索引](docs/实验与报告索引.md) · [数据与视频下载](docs/数据与发布索引.md)

**2026-09-11 新增阶段汇报：** [Walker Walk：980.91 / 977.50回报](DMControl_Walker_Walk_TD-MPC2复现与实验结果汇报.html) · [Pick Cube：9/10 / 10/10末步成功](ManiSkill2_Pick_Cube_TD-MPC2复现与实验结果汇报.html)。均为种子1/2各100万环境步终点，种子3未完成；含学习曲线、作者同预算对照与正式视频。[证据和使用说明](reports/two_task_replication_20260911/README.md)。

## 已有结果与研究进展

文档整理于 **2026-09-10**。下表依据已入库证据，不是服务器实时监控。

| 实验线 | 已有证据 | 可以得出的结论 |
|---|---|---|
| **Meta-World V2 / TD-MPC2 原方法** | 已完成 50 万环境步；最终末步成功 **8/10**；中途最好 9/10；约 4 小时 4 分钟 | 单任务、单训练种子的缩减预算复现；无教师示范或自研算法改进。[完整结果](tdmpc2_original/reports/50万步复现结果.md) |
| **ManiSkill-HAB / TD-MPC2 原方法迁移** | 50 万步协议、验证及 9 月 9 日启动证据已入库；正式完成证据尚未入库 | HAB 不是原论文报告的任务；启动记录不能作为最终成功率。[记录](hab_tdmpc2_original/reports/验证与启动记录.md) |
| 历史接触控制、短训、BC、示范辅助实验 | 各自有独立协议、结果或阶段记录 | 用于诊断与对照，不能与 V2 原方法复现混算。[分类索引](docs/实验与报告索引.md) |
| **下一阶段：候选动作后果排序** | 研究方案与 2025–2026 后续文献核验已整理 | 待验证研究假设；尚未证明创新或性能提升。[方案](refine-logs/FINAL_PROPOSAL.md) |

“末步成功”是在回合最后一步满足任务成功条件；“曾经成功”是中途至少成功一次，两者分开报告。Meta-World 的三个成功补录视频不计入正式 8/10 结果。当前没有世界模型提高 HAB 成功率的充分证据。

## 汇报与演示

| 想看什么 | GitHub 在线阅读 | 本地展示 |
|---|---|---|
| TD-MPC2 背景、创新、模块、规划、训练、评估及局限 | [算法流程报告](reports/TD-MPC2算法流程汇报.md) | [单文件 HTML](TD-MPC2算法流程汇报.html) |
| BMPC 背景、专家迭代、42项任务、原论文结果与复现核对 | [BMPC 论文汇报](reports/BMPC论文汇报.md) | [单文件 HTML](BMPC论文汇报.html) |
| Meta-World 复现协议、六阶段曲线、最终成绩及失败案例 | [实验汇报](reports/MetaWorld_TD-MPC2复现与实验结果汇报.md) | [汇报 HTML](MetaWorld_TD-MPC2复现与实验结果汇报.html) |
| 最终模型的视频演示 | [视频选择与指标边界](tdmpc2_original/reports/演示说明.md) | [离线演示包 v0.5.1](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.5.1-metaworld-method-report) |

GitHub 文件页显示 HTML 源码；下载或克隆后用浏览器打开。含视频的页面需要配套视频文件，推荐下载演示 ZIP、完整解压后打开其中的 HTML。仅查看报告和视频不需要 GPU。

## 复现入口

```bash
gh repo clone Pendragon08/hab-contact-aware-world-model
cd hab-contact-aware-world-model
```

训练需要 Linux、NVIDIA GPU 及对应仿真依赖；克隆代码不会自动下载场景资产、模型或全部实验数据。根据目标选择一个入口，先阅读协议并完成环境验证，再启动训练：

| 目标 | 环境、配置与命令 | 执行核验 |
|---|---|---|
| Meta-World V2 原方法复现 | [tdmpc2_original/README.md](tdmpc2_original/README.md) | [EXECUTION.md](tdmpc2_original/EXECUTION.md) |
| ManiSkill-HAB 原方法迁移 | [hab_tdmpc2_original/README.md](hab_tdmpc2_original/README.md) | [EXECUTION.md](hab_tdmpc2_original/EXECUTION.md) |
| 历史示范辅助 Meta-World V3 | [metaworld_exp/README.md](metaworld_exp/README.md) | 该目录协议与阶段报告 |
| 历史 HAB 接触控制及短训 | [复现说明](docs/复现说明.md) | [实验与报告索引](docs/实验与报告索引.md)中的对应验证记录 |

项目实验使用 GPU1（RTX A6000 48GB），GPU0 留给其他使用者并至少保留 12 GiB；具体入口有资源检查。无需调用 LLM API。本机路径与凭据保存在未入库的本地配置中。新机器的完整环境安装仍需实际验证。

## 数据与文件放在哪里

**Git 保存代码、配置、小型指标、报告和审计；Releases 按实验发布大文件。并非所有实验的全量原始数据都已上传。** 已核验的 8 个 Release、演示下载、归档恢复方法及待发布范围见 [数据与发布索引](docs/数据与发布索引.md)。

```text
tdmpc2_original/       Meta-World V2：官方 TD-MPC2 固定版本、运行入口与结果
hab_tdmpc2_original/   HAB：原方法的环境适配、运行协议与阶段证据
metaworld_exp/         Meta-World V3：历史示范辅助实验
scripts/ + configs/   历史 HAB 实验入口、配置、报告与归档工具
vendor_tdmpc2/         历史实验使用的算法源码与来源记录
tests/                控制、回放及归档等核验
reports/              汇报原稿、图表、指标与审计
docs/                 实验导航、复现说明与数据发布导航
refine-logs/          下一阶段研究方案、文献对照与历史审阅
history/              旧实验及 README 历史快照
```

根目录 HTML 保留稳定入口。下载恢复的 `artifacts/`、`release/` 及机器配置不进入 Git。

## 研究方案与来源

- [下一阶段研究方案：世界模型动作后果排序](refine-logs/FINAL_PROPOSAL.md)
- [五篇后续工作：共同任务、研究空白与对比边界](refine-logs/五篇TD-MPC2后续工作_共同任务与研究空白核验_20260910.md)
- [2026 方法与方案对应](refine-logs/2026方法来源与方案对应_20260910.md) · [ManiSkill 相关论文](docs/ManiSkill相关论文.md)
- TD-MPC2：[作者项目](https://www.tdmpc2.com/) · [固定版本源码](https://github.com/nicklashansen/tdmpc2/tree/57158282b46ebc5c329c5be9cfe2b0094126d1ca)；HAB：[官方代码](https://github.com/arth-shukla/mshab) · [官方权重](https://huggingface.co/arth-shukla/mshab_checkpoints)

README 作为人工维护的导航。旧细节见 [整理前快照](history/readme/README_20260910_before_cleanup.md)，后续新增结果优先写入对应实验报告，再更新首页摘要。评估时始终注明环境版本、预算、训练种子、评估回合及成功口径。
