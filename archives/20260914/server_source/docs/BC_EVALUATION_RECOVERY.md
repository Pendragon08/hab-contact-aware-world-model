# 固定检查点的评估续跑

100 次示范、53 条成功拟合轨迹及 5,000 更新已完成。第一次学生留出评估完整执行 200 步但全量起点哈希检查失败；该失败回合保留并计入最终 20 次评估。不会重新训练或重新抽取种子。

诊断：相同任务名称、机器人、活动物体、控制器与两路三帧深度输入相同。上游把 init-config 名称集合转换为进程局部编号；隐藏物体位于 x=y=-10000、z≈-10000 的场外停放区，其 z 槽随构建顺序变化。新校验只将 init_config_idxs 替换为相同任务计划中的语义名称，并规范化双方均在该停放区的物体 z 槽；其他物体状态、关节、控制器和观测逐项校验，保持 1e-5 容差，完整初始教师观测仍要求字节相同。

独立预检唯一调用，须原样执行一次，不修复、不重试、不执行 run：

```powershell
ssh -o BatchMode=yes ouyanghao@10.147.17.32 'bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/scripts/run_bc_eval_v2.sh witness'
```

预期退出 0、`BC_EVAL_CANONICAL_WITNESS_PASS`。预检先核验已完成的第一例学生失败起点；再重新执行教师在相同 seed 1210080、layout 6、task_idx 456、spawn_idx 26 的前 8 步，与原始 200 步教师轨迹的前缀比较状态、动作、深度、物理量和事件，误差≤1e-5。没有学生重试或模型更新。输出 `artifacts/bc_v1_2/eval_witness_v2`。

通过后把 witness 改为 run，仅续跑其余 19 个留出起点，固定 checkpoint_05000.pt 并核验其 SHA256 不变。继续使用 GPU1，最长 3,200 秒；与原训练和短预检合计仍在原 R003 一小时预算内。旧 process_train.json 失败状态保留，另生成过程汇总；原训练阶段 PyTorch 峰值未写出，标为不可得，整卡资源日志仍保留。门槛不变：曾抓取≥10/20、曾完整成功≥5/20。
