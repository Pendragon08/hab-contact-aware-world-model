# Walker Walk 独立入口验证

历史环境/学习入口验证文档。当前用户已将正式预算改为每种子100万环境步，见 BUDGET_EXECUTION.md；本文件短验证的3000决策协议保持不变，末尾400万步描述仅代表初始方案。

只执行下面的命令一次；不修改、不修复、不重复运行、不启动正式训练：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/dmc_tdmpc2_original/run.sh witness"
```

如果 SSH 返回仍在运行的会话，继续读取同一次会话直至结束，不要重发命令。

预期退出码0，末尾出现 `DMC_ORIGINAL_WITNESS_PASS`。此命令只使用GPU0，启动前要求GPU0无计算进程、使用显存小于500MiB且空闲至少12GiB、/data至少15GiB空闲。

真实环境检查：同种子官方DMControl封装和原生Walker Walk环境，500个固定动作，每个重复2步；观测、累积奖励最大差≤1e-10，完整回合500次决策，渲染384×384×3。

真实训练检查：独立随机初始化5M模型，3000动作决策/6000环境步，前2500动作的探索和原版初始化更新顺序，2999次优化器更新；开始和结束各评估完整500动作回合，共2000评估环境步。五个模块参数均改变，更新指标全为有限值；输出原始轨迹、两段评估视频、模型和 `complete.json`。

结果目录：服务器 `dmc_tdmpc2_original/artifacts/witness_<UTC时间>/`。末尾JSON给出结果；请另行只读读取本次 `complete.json`，核对预算、模块变化和环境误差。若不符合预期，原样报告，不要修复。

该短验证不代表Walker Walk性能达标。正式方案是3个种子，每个400万环境步，在同一GPU0顺序执行。
