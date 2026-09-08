# BC 固定策略诊断入口验证

本轮按 run-experiment 技能的独立文档预检要求，由新的 `bc_diagnosis_witness` 执行 [诊断说明](../docs/BC_DIAGNOSIS_EXECUTION.md) 中的命令一次。环境 d873b21f 原样复用；没有安装依赖、重试失败、启动正式阶段或修改文件。

- 独立执行退出码 0；输出 `BC_DIAGNOSIS_WITNESS_PASS`。
- seed 1210080、spawn 26、layout 6、task 456，8 步；不是新的留出成功率测试。
- GPU0 启动前空闲 48,532 MiB；GPU1 空闲 48,512 MiB，资源检查通过。
- 现场旁路教师建议与离线重建最大误差 1.4826655387878418e-05；原教师输出重建最大误差 1.6033649444580078e-05，均低于事先固定的 1e-4。
- 运行耗时 47.40 秒，PyTorch 峰值分配显存 765.28 MiB。

独立见证者指出：标准输出只打印两项无标签误差，没有枚举所有内部检查。主任务随后读取保存的 `artifacts/bc_diagnosis_v1/witness/complete.json`，确认学生前缀的 teacher_states、raw_actions13、executed_actions13、effective_actions11、states45、两路深度和 physical_states 最大差全部为 0.0，语义起点校验通过；逐步事件相等是入口必须通过的断言。这一枚举来自保存产物的复核，不冒称是独立见证者的逐字段复算。

冻结诊断配置 SHA256：`57cdfa38ccb2cd4612b8e5066d34b848c8843d6273d95d668f80a2661c6bc5d9`。短预检不替代20回合正式测试；教师旁路建议不执行，检查点不更新，原留出成绩不重试。
