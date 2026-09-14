# 固定行为克隆策略的失败诊断

环境沿用 `.aris/compute/ssh.md` 的 d873b21f，无依赖或原实验变动。仅 GPU1；GPU0 至少空闲 12 GiB。模型固定为 bc_v1_2 的第 5000 更新，SHA256 在 configs/bc_diagnosis_v1.json 中冻结。没有优化器或参数更新。

## 独立短预检

执行以下命令一次，不修改、不重试，也不启动 fit/offline：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/scripts/run_bc_diagnosis.sh witness"
```

预期退出 0，打印 BC_DIAGNOSIS_WITNESS_PASS。固定学生在原留出 seed 1210080 / spawn 26 执行 8 步；教师只旁路输出建议，不参与控制。学生轨迹前缀与原记录逐字段误差≤1e-5，事件相同、语义起点一致。保存的完整初始深度历史用于重建前两步；离线教师动作与现场旁路动作、原教师动作误差≤1e-4。原数据和检查点不能修改。预检上限300秒。

若失败，只记录失败和文档差异，交由主任务诊断；不改文件、不扩大容差、不清理失败目录。产物在 artifacts/bc_diagnosis_v1/witness、对应 process/provenance JSON 与 logs/bc_diagnosis_v1_witness*。

## 预检通过后的有限正式执行

顺序执行 `bash scripts/run_bc_diagnosis.sh fit`，再执行 `bash scripts/run_bc_diagnosis.sh offline`。各自上限 1200、900 秒，拒绝覆盖输出。不得在进程运行时修改启动文件或算法脚本。

fit：按冻结规则从53条成功拟合示范中等间距选20个起点，原模型每次完整运行200步，共4000新物理步。每步额外记录教师建议，但始终只执行学生动作；保存视频、完整深度、状态和初始快照，并核对与原教师起点一致。

offline：重用已有20条留出学生轨迹，分析20条拟合学生、20条留出学生及各自20条原教师轨迹，共80条。重建冻结教师在记录观测上的动作；原教师轨迹及现场旁路建议作为校验。比较运动/夹爪误差，以及相对10600个成功拟合状态的最近邻距离（只用前42维，按原训练统计标准化）。不推进记录轨迹的模拟，不产生反事实结果标签；环境初始化只用于加载原教师的输入结构。

这是诊断性样本：拟合组按教师结束成功筛选，留出组未这样筛选，不能将两组差异直接当成无偏泛化估计。最近邻距离只是状态偏离的代理指标，动作分歧不能单独证明因果。留出轨迹及补算教师标签禁止进入任何后续训练。主 HAB 成功判据不改；所有失败保留。不启动世界模型、风险规划或新的大规模训练。

本轮预估原始数据加压缩包不超过350MiB；本地D盘保留至少2GiB。报告和全部新增数据分别提交 Git 与新的 Release，既有 v0.3.0 数据不改。
