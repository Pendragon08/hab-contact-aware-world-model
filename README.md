# TD-MPC2 Reproduction and Contact-Aware World Models

<div align="center">

**世界模型与机器人控制：基线复现及探索实验归档**

[实验索引](docs/实验与报告索引.md) · [TD-MPC2 复现](tdmpc2_original/README.md) · [完整归档](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/server-archive-20260914) · [恢复说明](docs/服务器归档与恢复_20260914.md)

</div>

本仓库保存 TD-MPC2 在 Meta-World、DMControl、ManiSkill2 和 ManiSkill-HAB 上的复现与迁移实验，以及接触感知世界模型方向的探索记录。内容包括源代码、环境配置、训练日志、评价结果、模型归档和演示。

```text
环境观测 → 潜在状态编码 → 世界模型预测 → MPC 动作规划
        → 环境执行 → 经验回放 → 模型与价值学习
```

## Project Status

**项目已于 2026-09-14 归档，原服务器项目目录已在备份校验后清理。** 本仓库用于阅读、检查和按需恢复，不代表仍有训练队列运行。

- [x] 保存已完成的基线结果及未完成、未取得收益的尝试。
- [x] 完整服务器快照已上传 Release，29 个分片通过校验。
- [x] 提供逐文件清单、恢复工具和可浏览源码快照。
- [ ] 接触感知机制的完整有效性结论尚未建立。

## Table of Contents

1. [Setup](#setup)
2. [Phase 1: Restore Experimental Assets](#phase-1-restore-experimental-assets)
3. [Phase 2: Reproduce a Baseline](#phase-2-reproduce-a-baseline)
4. [Phase 3: Evaluation](#phase-3-evaluation)
5. [Results](#results)
6. [Repository Structure](#repository-structure)
7. [Documentation](#documentation)
8. [Acknowledgements and License](#acknowledgements-and-license)

## Setup

阅读代码与报告只需克隆仓库：

```bash
git clone https://github.com/Pendragon08/hab-contact-aware-world-model.git
cd hab-contact-aware-world-model
```

运行实验需要 Linux 和对应任务环境。各任务使用不同的仿真器与依赖，不能用一个通用环境替代：

| 实验 | 环境与安装入口 |
|---|---|
| TD-MPC2 / Meta-World V2 | [任务 README](tdmpc2_original/README.md) · [环境配置](tdmpc2_original/configs/env-spec.json) |
| TD-MPC2 / ManiSkill-HAB | [任务 README](hab_tdmpc2_original/README.md) · [环境配置](hab_tdmpc2_original/configs/env-spec.json) |
| DMControl / ManiSkill2 | [双任务复现报告](reports/two_task_replication_20260911/README.md) · [服务器源码快照](archives/20260914/server_source/) |

归档保存实验内容，不是整台服务器镜像；共享解释器、驱动及外部场景资产仍需另行配置。旧脚本中的路径与 GPU 设置应先适配，历史调度器不应作为自动恢复入口。

## Phase 1: Restore Experimental Assets

无需恢复归档即可阅读大部分源码和结果。需要完整模型、原始运行记录或轨迹时，再按需下载 [服务器归档 Release](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/server-archive-20260914)：

```bash
gh release download server-archive-20260914 \
  --repo Pendragon08/hab-contact-aware-world-model \
  --dir tdmpc2-backup \
  --pattern 'tdmpc2-server-20260914.tar.gz.part*' \
  --pattern 'archive_index.json' \
  --pattern 'source_manifest.json'

python scripts/restore_server_archive.py --archive-dir tdmpc2-backup --verify
```

恢复目标应是新建的独立目录。目录要求、完整性检查及共享环境说明见 [归档与恢复文档](docs/服务器归档与恢复_20260914.md)。该文档里的“私有仓库”描述的是归档时状态；仓库现已公开。

## Phase 2: Reproduce a Baseline

建议先阅读一个任务的冻结协议，再配置环境。以 Meta-World Pick-Place V2 为例，本仓库使用固定版本 TD-MPC2、5M 模型和 500,000 环境步预算，属于缩减范围复现。

在仓库根目录配置本地 `local-settings.sh` 中的 `HAB_PYTHON`，完成路径和资源检查后：

```bash
bash tdmpc2_original/setup.sh
bash tdmpc2_original/run.sh witness
# 环境验证通过后，按任务文档在新的实验目录运行
bash tdmpc2_original/run.sh train
```

完整过程见 [任务说明](tdmpc2_original/README.md) 与 [执行协议](tdmpc2_original/EXECUTION.md)。其他任务沿用各自入口；不把 ManiSkill-HAB 的迁移实验称为原论文已有的 HAB 结果。

## Phase 3: Evaluation

每项实验分别保存环境步数、种子、模型选择规则、评价回合、指标和完成记录。

- Meta-World：报告最后一步成功率，演示补录不计入正式分母。
- DMControl：报告回合回报，与成功率分别解释。
- ManiSkill2：按任务定义评价抓取、抬升等成功条件。
- ManiSkill-HAB：检查抓取、末端休止、机器人休止、静止与累计接触限制的联合条件。

训练损失下降、脚本正常退出及演示视频分别是过程证据，不能替代任务成功指标。评价口径见 [实验索引](docs/实验与报告索引.md)。

## Results

| 任务 | 已归档结果 | 解释范围 |
|---|---|---|
| Meta-World Pick-Place V2 | 最终评价 8/10 成功；500,000 环境步 | 单任务、单种子的缩减预算复现 |
| DMControl Walker Walk | seed 1 / 2：平均回报 980.91 / 977.50 | 每种子 1,000,000 环境步；不代表全部计划种子 |
| ManiSkill2 Pick Cube | seed 1 / 2：9/10、10/10 成功 | 对应归档评价回合，非完整论文基准 |
| ManiSkill-HAB Pick | 停止前 400,000 步评价 0/20 | 未完成原预算；保留失败记录 |

来源：[Meta-World 结果](tdmpc2_original/reports/50万步复现结果.md)、[DMControl / ManiSkill2 结果](reports/two_task_replication_20260911/README.md)、[实验总索引](docs/实验与报告索引.md)。扩展任务及种子的最终状态以 [服务器快照](archives/20260914/server_source/) 和完整 Release 为准。

## Repository Structure

```text
tdmpc2_original/       TD-MPC2 / Meta-World V2 基线
hab_tdmpc2_original/   TD-MPC2 / ManiSkill-HAB 迁移
metaworld_exp/         历史 Meta-World 探索实验
configs/              环境与实验配置
scripts/              数据处理、训练、评价及归档工具
reports/              结果报告、审计与复现分析
docs/                 方法说明、实验索引与恢复指南
archives/20260914/    已归档服务器源码与校验记录
vendor_tdmpc2/        上游代码及来源说明
```

## Documentation

- [TD-MPC2 算法流程](reports/TD-MPC2算法流程汇报.md)
- [实验与报告索引](docs/实验与报告索引.md)
- [数据与发布索引](docs/数据与发布索引.md)
- [复现说明](docs/复现说明.md)
- [整理前的完整 README](https://github.com/Pendragon08/hab-contact-aware-world-model/blob/c14bb27ca94ed9e8ca0d107527d795327da13374/README.md)

## Acknowledgements and License

本项目基于 [TD-MPC2](https://github.com/nicklashansen/tdmpc2) 与相关仿真任务展开复现，ManiSkill-HAB 的上游实现见 [mshab](https://github.com/arth-shukla/mshab)。固定源码版本与改动边界见 [来源记录](vendor_tdmpc2/PROVENANCE.md) 和各任务文档。

TD-MPC2 的上游许可证保留于 [LICENSE](tdmpc2_original/upstream/LICENSE)。仓库目前未提供统一的顶层 LICENSE，其他代码、数据与场景遵循各自条款。引用本仓库结果时，请注明任务、预算、种子和提交版本，并引用原方法。
