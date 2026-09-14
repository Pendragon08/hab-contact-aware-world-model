# ManiSkill2 Pick Cube 启动记录

核验时间：2026-09-10 17:10:24（北京时间）。

按用户最新要求，新ManiSkill2在GPU1训练，GPU0的DMControl保持运行。旧HAB任务已停止，历史日志与模型保留。

新任务：TD-MPC2官方方法、ManiSkill2 0.4.1/SAPIEN2.2.1的PickCube-v0，种子1/2/3顺序执行，每个种子100万原生环境步（50万决策，每动作重复2步）。同预算作者结果和运行时差异见上级README。

- 启动时间：17:09:03；screen：`1795149.tdmpc2_ms2_pickcube_3seed_1m_gpu1`。
- 训练进程：1795158；启动入口：`bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/maniskill2_tdmpc2_original/launch.sh`。已提交，不要重复执行。
- GPU1当前实测计算与图形显存862MiB；SAPIEN显式选择pci:65。GPU0仅额外保留4MiB Vulkan枚举图形上下文，无新CUDA计算任务。
- 首个种子已记录2002训练环境步、1000次真实优化器更新，loss有限。当前仅有未训练模型初始评估，成功率0不能当作最终结果。
- 独立验证已通过：2400环境步/1199更新，五个模块实际改变，12训练轨迹/2评估轨迹/2视频。独立报告与根代理复核结果见independent-witness.json、witness-complete.json。
- GPU0原DMControl进程1789362连续运行，最新记录99802环境步；原screen1789353仍存活，冻结源码哈希未变化。最近训练回报970.95仅是训练回报，不冒充评估成绩。

正式日志：服务器本目录`logs/seed1_1m.log`，轨迹/模型位于`artifacts/pick_cube_seed1_1m/`。每10万环境步评估10回合，报告末步成功率与success-once；每个种子完整结束后归档，三种子结束后汇总。初始热身吞吐不能可靠外推总耗时，待持续训练后估计。

精确进程、GPU、更新和初始评估证据见launch-snapshot.json。队列已在后台运行；尚未完成100万步，也没有最终性能结论。
