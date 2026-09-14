# 同时运行四个TD-MPC2任务

2026-09-11用户要求在显存允许时增加其他实验。采用已经实测的每GPU最多2个训练进程，任务分配如下：

| GPU | 原任务（继续） | 新任务 |
|---|---|---|
| 0 | Walker Walk | Walker Run（walker-run） |
| 1 | Pick Cube | Lift Cube（LiftCube-v0/lift-cube） |

新任务均为TD-MPC2原论文覆盖的任务，各3个种子，每个种子100万原生训练环境步=50万动作决策。每10万原生步评估10回合，评估交互另计。5M状态输入模型、batch256、horizon3，保留官方模型/优化器/损失/回放与更新顺序。新任务各自顺序完成3个种子；每张卡的两类任务互相并行。预算缩短及历史Python/PyTorch运行时差异沿用既有披露，不能称为原论文完整4M预算复现。

Walker Run使用24维状态/6维动作，回报为主要指标；Lift Cube使用42维状态/4维动作、原生dense奖励及末步success判据。环境和学习核心不改。GPU1使用pci:65渲染，旧Vulkan仍可能在GPU0保留每进程4MiB纯图形上下文。

作者同100万步参考：Walker Run种子1/2/3回报837.0/833.7/856.3；Lift Cube均100%末步成功率。这是作者参考，不是本轮实验结果。原始CSV保存在各任务reference目录，来源为官方固定commit57158282b46ebc5c329c5be9cfe2b0094126d1ca的results/walker-run.csv和results/lift-cube.csv。

新任务使用各自的walker_run、lift_cube目录，依赖只读复用原实验环境，临时文件/缓存/日志/轨迹/模型独立存储。源文件、启动入口及配置写入哈希证据。数据盘至少保留15GiB，两卡各保留12GiB显存；每任务完成后生成服务器归档与校验清单，失败停止对应任务后续种子。

## 原实验接续

原队列在启动下一种子时要求GPU空闲，直接加任务会导致其下一种子启动失败。为此manager只暂停原队列的父shell，当前Python训练子进程保持运行；等待其完整完成后归档，结束旧shell，再通过shared_entry接续剩余种子。原runner/runtime/config与当前训练进程不改、不重启。共享入口仅调整获授权的同卡进程检查，模型和训练步骤不变。已用独立的临时父/子进程验证暂停父shell不会暂停子进程，并且退役旧shell不会重复启动下一种子。

新任务依赖独立短验证通过后才能正式启动。验证入口为validate.sh；正式入口为launch.sh，均有防止重复提交的检查。正式screen为tdmpc2_four_tasks_1m，所有任务的调度状态以state.json为准，旧队列日志可能停留在交接前。handoff.json记录旧父/子PID及进程启动标识。不要手工恢复被交接的旧队列，以免重复启动。

## 查看结果

新任务日志：walker_run/logs/seed{1,2,3}_1m.log、lift_cube/logs/seed{1,2,3}_1m.log；结果位于相应artifacts目录。原任务数据继续位于原项目目录。若需要检查当前全部进程，应同时匹配runner.py和shared_entry.py，或直接读取state.json。

每个新增种子耗时初步按约10–11小时量级估计，具体取决于任务和同卡负载；已有实验的完工时间也会因共享计算资源略有延长。此前吞吐测试约1.9倍增益来自Walker Walk/Pick Cube组合，不保证新组合完全相同，正式启动后需检查实际更新和GPU资源。
