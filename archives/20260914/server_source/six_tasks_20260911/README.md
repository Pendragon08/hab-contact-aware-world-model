# 六任务并发训练

2026-09-11 用户要求在四任务基础上继续新增任务。本轮扩至每卡三个学习进程：

| GPU | 保留的任务 | 新增任务 |
|---|---|---|
| 0 | Walker Walk、Walker Run | Cheetah Run |
| 1 | Pick Cube、Lift Cube | Stack Cube |

新增任务各使用种子 1/2/3，每种子 100 万原生训练环境步，等于动作重复 2 下的 50 万决策步。每 10 万环境步评估 10 回合，评估交互另计。各任务种子顺序运行，各任务之间并行。继续使用官方固定版本 TD-MPC2 的 5M 状态输入模型、batch256、horizon3 和原有优化器、损失及更新顺序。保留此前披露的历史运行时差异及缩短预算差异。

Cheetah Run 使用原生 cheetah/run，17 维观测、6 维动作，500 决策步/回合，按回报报告。Stack Cube 使用原生 StackCube-v0，55 维观测、4 维动作，100 决策步/回合，dense 奖励及 pd_ee_delta_pos 控制；成功率按原生末步 success 判定，success-once 单列。未修改环境的奖励或成功条件。模型训练均绑定 GPU；沿用这些历史仿真环境自身的物理执行方式。

环境只读复用既有已验证依赖，没有重新安装。新入口仅把同卡进程检查从两个扩为三个，并验证所属用户、任务和 GPU；至少保留 12 GiB 显存和 15 GiB 磁盘。新增任务的缓存、日志、轨迹、视频、模型和服务器归档独立保存。作者固定 commit 的同任务 CSV 保存在各自 reference 目录；它们是参考数据，不是本轮结果。

六任务调度器 `manager.py` 只接管调度，不停止或重启正在训练的四个学习进程。先保存旧调度状态和进程身份，再结束旧调度器自身；逐个等待当前种子自然完成、归档，并通过新的三进程入口接续剩余种子。原 runner、runtime、run.sh、配置及四任务入口文件均不修改。旧调度器暂停/退役后，其旧 state.json 不再是最新状态，统一以本目录 state.json 为准。不要手动恢复已移交的旧队列。

短验证期间，旧调度器自身暂时 SIGSTOP，以防 Walker Walk 自然换种子时旧“两进程”启动检查与第三个验证进程冲突；所有学习进程照常运行。此操作单独记录在 validation-scheduler-hold.json。独立执行者按 EXECUTION.md 只运行一次验证，报告记录在 independent-validation.json。CPU 临时进程集成测试 handoff-smoke/result.json 检验替换调度器后四个子进程继续前进。短验证不作为正式性能结果。

正式启动入口是 launch.sh，screen 为 tdmpc2_six_tasks_1m；每个任务完成种子后保留原始数据并生成校验清单及服务器归档。某任务失败时保留尝试并停止其后续种子，不自动重试。启动器防止重复提交。

只读查看六任务状态：

```
python3 /data/code/ouyanghao/oyh/hab-contact-aware-world-model/six_tasks_20260911/status.py
```

吞吐测量采用新增前、进入正式训练后的各五分钟墙钟窗口，原始数据在 measurements，比较在 throughput-comparison.json。比较包含评估和记录开销，日志计数有 200 环境步量化；任务组合与种子阶段不同，属于本次实际总处理速度比较，不能推导算法学习效率，也不能宣称三进程是并发上限。六进程短验证时实测有温控降频，显存空余不能代表计算容量空余。
