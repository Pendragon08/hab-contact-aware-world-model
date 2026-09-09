# HAB 原方法迁移：独立入口验证

只执行以下命令一次，不修复、不重试、不启动正式训练：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/hab_tdmpc2_original/run.sh witness"
```

预期：退出0，输出 `HAB_ORIGINAL_WITNESS_PASS`。真实 HAB 环境执行1,200训练步，使用官方 Buffer、5M网络和官方更新顺序；开头1,000步随机探索后执行原版seed-buffer更新，随后在线更新。开始和结束各评估一个完整200步回合并保存视频，共400评估步；五个模型模块参数均改变，所有指标有限，成功判据与原五条件合取一致。结果目录 `hab_tdmpc2_original/artifacts/witness_<timestamp>/`，包含 `complete.json`、metrics、逐回合轨迹/事件和视频。

本次短验证允许与本项目正在运行的 Meta-World 原版训练共享 GPU1。入口必须检查 GPU1 上已有计算进程仅属于同一用户的指定 `tdmpc2_original/runner.py train`，并检查两卡空闲显存均≥12 GiB、磁盘空闲≥15 GiB；不满足就拒绝运行。不得停止已有进程。正式HAB训练将串行排队，不能与Meta-World正式训练同时启动。共享条件下见证速度不能当作独占GPU基准。

运行环境只读复用HAB已验证环境，无新依赖安装；原版TD-MPC2来源固定于 `57158282b46ebc5c329c5be9cfe2b0094126d1ca`。输入44维（原生42+瞬时力/100+累计力/5000，无时间特征），动作11个有效通道，头部控制为零；这些是任务接口约定。学习损失和MPC无额外创新。HAB不是原论文报告过的任务。

如与文档不符，原样报告错误、退出码和输出；不要自行更改或重复运行。
