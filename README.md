# HAB Contact-Aware World Model

面向 ManiSkill-HAB 移动机械臂抓取的接触风险控制与世界模型规划研究。

机器人为 Fetch（移动底盘、升降躯干和机械臂），当前任务为 TidyHouse Pick 子任务。研究过程从官方 SAC 的可复现对照实验开始，再验证状态输入的 TD-MPC2 适配与训练可行性。尚未获得世界模型改善 HAB 成功率的实验结论。

## 实验状态

|阶段|状态|
|---|---|
|已有官方 SAC 开发评估：5 个家具布局 × 20 回合|完成；曾经成功 66/100，结束成功 52/100|
|原始 SAC、保守控制、接触力规则控制的配对实验|独立验证通过；GPU1 正在执行 300 回合|
|TD-MPC2 状态输入适配与短程训练|独立验证通过；正在执行每算法 10,000 步 pilot|
|学习风险模块及独立测试|依赖前置基线结果，未训练|

已有评估的五个布局来自同一个验证宏观场景，不能称为五个独立房屋。全部开发评估与最终测试严格区分。14 个曾成功、结束未成功的回合在最后仍有抓取信号，不能当作 14 次掉落。

## 仓库结构

- `scripts/`：实验、指标审计、报告和归档代码。
- `configs/`：预先固定的实验方案、软件版本、随机种子。
- `tests/`：控制通道、因果输入和结果完整性检查。
- `reports/`：中文实验报告与机器可读汇总。
- `docs/`：复现、算法、数据来源和限制说明。
- GitHub Releases：完整实验轨迹、初始状态、模型检查点、视频和原始日志，附 SHA256 清单。大文件不直接塞入 Git 历史。

已提供 [固定实验方案](docs/实验方案.md)、[复现说明](docs/复现说明.md)、[控制入口验证](reports/入口验证.md)和[状态训练入口验证](reports/状态训练入口验证.md)。完整结果将在所有回合与审计完成后写入本文件。当前不应把未完成阶段当作已经复现。

原有 100 回合与历史见证已归档在 [v0.1.0-baseline Release](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.1.0-baseline)，六个附件的 GitHub SHA256 均已核对。

## 数据与上游

- [ManiSkill-HAB / ICLR 2025](https://github.com/arth-shukla/mshab)
- [ManiSkill-HAB 任务、数据和评估说明](https://arth-shukla.github.io/mshab/)
- [官方 SAC 检查点](https://huggingface.co/arth-shukla/mshab_checkpoints)
- [ManiSkill](https://github.com/haosulab/ManiSkill)
- [TD-MPC2 / ICLR 2024](https://www.tdmpc2.com/)

本项目生成的实验数据将随 Release 保存。第三方场景资产与权重保留来源、版本、下载方式和校验值，按其上游许可使用；不能把第三方资产声明为本项目自有数据。

## 计算资源

使用一张 RTX A6000 48GB（GPU1）；GPU0 留给其他使用者。运行前检查资源，不终止他人任务。无需 LLM API。所有训练耗时以实际测量为准。
