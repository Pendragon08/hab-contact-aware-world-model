# BC 双观测入口诊断

本次诊断发生于正式 R001–R003 前。首次独立文档预检退出 1，baseline 在 train 布局 6、种子 1199999、task_idx 380 完成 8 步；dual 在初始状态一致性断言处停止，没有启动正式示范采集或训练。

逐字段读取验证发现：教师保存的 42 维状态和通过官方原生状态路径取得的前 42 维顺序相同。差异只在目标物体的相对四元数分量，最大绝对误差 5.960464477539063e-8，位置、关节状态等其余值一致。原入口使用 `torch.equal` 要求逐位相等，比冻结计划的 1e-5 数值容差更严格。

修正版本 bc_v1_1 使用原定绝对容差 1e-5；没有改变数据分割、任务奖励、物理步数、成功条件或通过门槛。原生状态路径另以 `get_obs(info)` 对照验证；同一读取前后的环境、控制器和 PyTorch/CUDA 随机状态摘要相同。

- 原失败记录：`artifacts/bc_v1/witness/` 和 `process_witness.json`。
- 原入口源码及配置：`artifacts/bc_v1/source_snapshot/`。
- 独立诊断原始值：`artifacts/bc_v1/state_diagnosis/diagnosis.json` 与 `reads.pt`。
- 修正后入口及验收方式：[BC 执行说明](../docs/BC_EXECUTION.md)。

这些记录均须随新 Release 归档。短预检通过也不替代 20 组完整配对验证。
