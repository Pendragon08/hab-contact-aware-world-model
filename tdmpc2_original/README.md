# TD-MPC2 原方法：Meta-World Pick-Place V2

[返回项目首页](../README.md) · [实验索引](../docs/实验与报告索引.md)。本轮已于 2026-09-09 完成，最终末步成功 8/10；[完整结果及归档状态](reports/50万步复现结果.md)。下文保留本轮实际执行协议。

算法解释：[算法流程、训练评估与复现边界](reports/算法流程与复现边界.md)。当前 V2 结果属于原方法复现，尚无自研算法改进。

演示已导出：打开 [最终模型演示页面](../MetaWorld原版世界模型演示.html)，或下载 [离线演示包](https://github.com/Pendragon08/hab-contact-aware-world-model/releases/tag/v0.5.0-metaworld-original-demo)。含三个固定种子的成功补录及原始失败视频；补录不计入正式 8/10 成绩。


这是**缩减预算的原方法复现**，不是完整论文结果复现，也不是先前加入示范数据的 V3 实验。

## 本轮协议

| 项目 | 设定 |
|---|---|
| 官方 TD-MPC2 | `57158282b46ebc5c329c5be9cfe2b0094126d1ca`，ICLR 2024 时期 |
| Meta-World | `04be337a12305e393c0caf0cbf5ec7755c7c8feb`，官方依赖指定的 V2 |
| 任务 | `pick-place-v2-goal-observable` |
| 原始观测 / 动作 | 39 / 4，无额外时间特征、无观测归一化 |
| 网络 | 官方 5M 配置，从随机权重开始 |
| 预算 | 250,000 次动作决策 × action repeat 2 = 500,000 环境步 |
| 回合 | 100 次决策，即 200 环境步 |
| 初始探索 | 原版 1,000 决策随机采样及随后 1,000 次 seed-buffer 更新 |
| 学习和规划 | 官方损失、经验回放、策略先验、Q 集成和 MPC，折扣由原公式得 0.95 |
| 评估 | 0/10/20/30/40/50 万环境步，各 10 回合，主指标为最后一步 success |
| 训练种子 | 第一轮 seed=1；不是三种子论文统计 |
| 资源 | 仅 GPU1；GPU0 不占用，保留至少 12 GiB |

不使用教师示范、行为克隆损失、纠正数据、接触规则或风险惩罚。这里的初始 seed-buffer 更新仍属于原论文在线训练过程，不是加载预训练权重。

### 与原实现的差异及边界

- `upstream/` 所选文件保留官方原文，启动和结束均验证 SHA256；包含 MIT 许可。
- `runner.py` 复用官方 `OnlineTrainer.train`，只把预算循环 `<=` 改为 `<`，避免多执行一次动作；结束后显式调用原版评估。没有重写学习顺序。
- 本地日志、原始轨迹、视频及权重保存替代 W&B；原始评估仍使用相同环境和随机序列，不喂回评估数据。
- 原版 reset 内部还有一次零动作 step，保留该行为。50 万步不含 reset 的附加步，附加步和评估步在完成记录里单列。
- 使用服务器只读 Python 3.10 / PyTorch 2.5.1 / TorchRL 0.6 基础依赖，而不是完全复制当年的 Python 3.9 / nightly TorchRL；必须通过真实 replay、优化器、物理仿真和渲染验证。
- MuJoCo 为原生 2.1.0 / mujoco-py 2.1.2.14。为现代驱动目录添加**本项目安装副本**的库路径发现补丁，记录前后哈希，不改物理和任务。
- 只有一个任务、一个种子，不能据此声称复现论文全套结果或获得稳定改进。首轮结果出来后再决定多种子或预算扩展。

## 安装与执行

在仓库根目录配置本地未入库的 `local-settings.sh`，指定只读 `HAB_PYTHON`。按顺序运行：

```bash
bash tdmpc2_original/setup.sh
bash tdmpc2_original/run.sh witness
bash tdmpc2_original/run.sh train
```

安装必须先有 `configs/env-spec.json`。启动会检查 GPU1 占用，达到 500 MiB 就拒绝启动；不会终止已有任务。不要覆盖旧结果目录。正式训练通过专用 screen 会话运行。

独立执行验证说明见 `EXECUTION.md`。验证通过后才将环境标为可用。

## 输出

- `artifacts/seed1_500k/resolved-config.yaml`：完整实际配置。
- `metrics.jsonl`：训练、评估日志及明确的决策步/环境步计数。
- `trajectories/{train,eval}/*.npz`：全部观测、动作、奖励及成功标记。
- `models/step_*.pt`、`final.pt`：评估时及最终权重。
- `videos/eval_*.mp4`：每次评估第一回合视频。
- `complete.json`：仅全部训练和终点评估成功后生成。

训练数据保存在服务器。完成后用仓库 `scripts/archive_artifacts.py` 打包，经校验后上传 GitHub Release。代码和协议入 Git；不能把尚未传回或上传的数据写成已备份。

## 原始来源

- [论文 PDF](https://arxiv.org/pdf/2310.16828v2)
- [固定版本官方代码](https://github.com/nicklashansen/tdmpc2/tree/57158282b46ebc5c329c5be9cfe2b0094126d1ca)
- [官方环境依赖](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/docker/environment.yaml)
- [Meta-World 固定版本](https://github.com/Farama-Foundation/Metaworld/tree/04be337a12305e393c0caf0cbf5ec7755c7c8feb)
- [官方 Pick-Place 原始曲线](https://github.com/nicklashansen/tdmpc2/blob/57158282b46ebc5c329c5be9cfe2b0094126d1ca/results/mw-pick-place.csv)

本文件记录执行协议；完成情况、评估数据及全量归档交付范围以 [结果报告](reports/50万步复现结果.md) 为准。
