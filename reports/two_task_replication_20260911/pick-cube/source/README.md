# TD-MPC2 原论文 ManiSkill2 Pick Cube：100万步复现

2026-09-10用户不再使用HAB任务，明确要求新ManiSkill2任务使用GPU1，GPU0的DMControl继续运行。原HAB于北京时间16:47:21停止，最后已写训练记录为426701步，原始记录和模型保留，没有冒充50万步训练完成。

选择原论文直接报告的 `pick-cube` / `PickCube-v0`。使用ManiSkill2 0.4.1、SAPIEN2.2.1、Gym0.21和NumPy1.23.5；不是ManiSkill3，也不是HAB场景。Panda机械臂，官方51维状态、4维动作、pd_ee_delta_pos控制、原生dense奖励和原生成功条件。模型与训练核心固定于官方TD-MPC2 commit `57158282b46ebc5c329c5be9cfe2b0094126d1ca`，逐文件哈希核验。

## 当前方案

- 种子1/2/3在GPU1顺序运行，各100万环境步=50万动作决策，每动作重复2步。
- 每回合100次决策=200环境步，忽略中途成功终止，主指标为最后一步的原生success；另报success-once。
- 官方5M配置、batch256、规划horizon3、原版Buffer/优化器/损失/更新顺序；1000决策探索，折扣按官方公式得到0.95。
- 每5万决策/10万环境步评估10回合，共11轮含初始和最终；记录所有训练/评估状态、动作、奖励、物理状态和成功标志，保存每轮首回合视频。
- 与作者同为100万步的CSV比较：seed1=100%、seed2=80%、seed3=100%。这是作者参照，不是本轮结果。原论文正式训练4M，本轮只能称缩减预算复现。

来源：[论文附录C/D](https://arxiv.org/html/2310.16828v2)、[作者pick-cube原始结果](https://raw.githubusercontent.com/nicklashansen/tdmpc2/57158282b46ebc5c329c5be9cfe2b0094126d1ca/results/pick-cube.csv)。副本存于reference/pick-cube.csv。

## 工程适配与证据边界

Python3.10/PyTorch2.5.1/TorchRL0.6沿用已经验证的现有运行时，与论文历史运行时不同。官方训练循环仅将<=改为<以精确满足预算，最终显式评估；显式固定Gym动作空间随机种子。官方ManiSkill2封装不改物理过程、观测、动作、奖励和判据，构造时仅注入渲染设备参数以绑定GPU1；录像直接调用原生render(mode='cameras')。所有适配在独立入口内，上游文件只读。

GPU0上DMControl文件、进程和依赖不改动。新依赖只安装于本目录vendor_deps；只读复用现有NumPy/Gym与解释器。GPU1启动要求无计算任务、显存用量小于500MiB；GPU0空闲至少12GiB、数据盘至少15GiB。正式启动前必须通过真实环境与优化器验证和独立文档执行。

## 输出

GPU绑定使用`CUDA_VISIBLE_DEVICES=1`及SAPIEN `device=pci:65`。实测Vulkan初始化会在GPU0保留4MiB纯图形上下文；正式仿真渲染与CUDA计算在GPU1。验证禁止GPU0计算上下文，并限制该图形上下文不超过8MiB。GPU0上的DMControl进程与源码哈希已另行核验。

`artifacts/pick_cube_seed{1,2,3}_1m/`；日志 `logs/seed{1,2,3}_1m.log`；专属screen见launch.json。每种子完整结束生成complete.json，然后服务器归档和SHA256清单。三种子完成后报告均值，不选最好种子替代全部结果。任何失败停队列，保留现场，不自动重跑。

只有完成标记中的train_environment_steps=1000000且末轮评估完成，才能宣布该种子训练完成；启动或验证通过不代表成功率达标。服务器归档不等于已经回传或发布。
