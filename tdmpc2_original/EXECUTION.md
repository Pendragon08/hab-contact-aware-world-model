# 独立执行验证

服务器环境的 declarative spec：`configs/env-spec.json`。

从本机执行以下命令，严格执行一次，不修改、不自行重试：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/tdmpc2_original/run.sh witness"
```

预期退出码 0，末尾 `ORIGINAL_WITNESS_PASS`。程序运行原版 5M 网络、原版 Buffer：300 决策随机经验，5 次真实优化器更新，12 次 MPC 决策并输出 MP4；另外两次相同种子的 20 决策物理回放。五个模块参数都必须改变，采样形状为 4×256×39，所有损失有限，两次重复误差 ≤1e-6。仅 GPU1，启动前显存占用必须小于 500 MiB；若 GPU 正忙应退出 20，不能结束别人的任务。

结果写入服务器 `tdmpc2_original/artifacts/witness_<timestamp>/witness.json`，包括原始损失、模块变更、显存和重复误差。视频为 `videos/eval_000000.mp4`。

这是环境及方法入口验证，不是任务成功验证。任何安装问题、连接失败或预期与实际不符都如实报告，保留失败输出。正式训练不是本次独立验证的工作范围。
