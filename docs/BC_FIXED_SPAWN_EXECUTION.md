# 固定起点分割的学习入口（bc_v1_2）

R001 使用 bc_v1_1 的 20 组完整教师配对结果；本协议只接续 R002/R003。公开 train spawn 数据中每个任务有 100 条起点。不同随机种子不保证起点不同，因此在采集前使用 Python Random(1210000) 打乱起点编号 0–99，前 80 为拟合、后 20 为开发留出。完整排列已冻结在 `configs/bc_v1_2.json`。每一行只采集一次；失败也保留。任务仍按 R001 第一条教师结束成功的候选选取。

旧 bc_v1_1 尚未采集 R002；因此本修正不是看过开发成绩后改变分割。种子、训练步数、5,000 更新与成功门槛保持原计划，追加起点编号隔离和拟合/留出初始状态去重检查。

## 独立短预检：唯一调用

沿用 `.aris/compute/ssh.md` 的 d873b21f 环境。仅原样执行下列调用一次，不修改或重试：

```powershell
ssh -o BatchMode=yes ouyanghao@10.147.17.32 'HAB_BC_CONFIG=configs/bc_v1_2.json bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/scripts/run_bc.sh witness'
```

预期退出 0 并打印 `BC_DOC_WITNESS_PASS`。实际在 GPU1 上运行布局 6、种子 1199999、明确固定 spawn_idx=0 的两次 8 步 GPU 模拟，对比状态/深度/动作/成功条件，误差≤1e-5；读取原生状态与一次 BC 前向/反向更新。日志为 `logs/bc_v1_2_witness.log`，数据为 `artifacts/bc_v1_2/witness`。预检不是训练数据。

## 正式运行

R001 完成且本预检通过后，将调用最后的 `witness` 依次改为 `collect`、`train`。入口对 collect 设 2 小时上限、train 设 1 小时上限；GPU0 不使用、保留至少 12 GiB，GPU1 共享占用冲突时停止，不终止他人进程。不运行 v1_2 的 interface；它明确读取 `artifacts/bc_v1_1/interface/complete.json` 和同版本 `selection.json`。

完整轨迹为 201 个状态、200 个动作；归一化和拟合仅使用前 80 起点内结束成功的示范（至少十条）。留出 20 个起点从不进入优化器或归一化器。固定第 5,000 次更新进行闭环评估，不按开发成绩挑模型。学生评估使用与教师完全相同的任务、起点编号和随机种子；同起点快照须匹配。门槛为曾抓取≥10/20、曾完整成功≥5/20，另报结束完整成功。任何失败均记录并停止后续扩展。

审计命令：在加载项目 runtime 后，用 `HAB_BC_CONFIG=configs/bc_v1_2.json python scripts/report_bc.py`。审计包括训练/开发起点隔离、数据和动作对齐、5,000 更新、最终动作重算、完整视频和成功判据，不修改成绩。
