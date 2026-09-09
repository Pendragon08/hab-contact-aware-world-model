# 示范辅助 TD-MPC2 独立入口验证

独立见证者：fresh agent `demo_tdmpc2_witness`。日期：2026-09-09。仅阅读 run-experiment/SKILL.md、.aris/compute/ssh.md、docs/DEMO_TDMPC2_EXECUTION.md，完成后读取见证 process.json / complete.json；未读取实现、修改代码、修复、重跑或启动正式训练。

## 执行与结论

文档命令原样执行一次：

```text
ssh -o BatchMode=yes -o ConnectTimeout=10 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/scripts/run_demo_tdmpc2.sh witness"
```

SSH进程退出码 **0**。终端最终原文：

```text
DEMO_TDMPC2_WITNESS_PASS {'encoder': True, 'dynamics': True, 'reward': True, 'Q': True, 'policy': True} 0.0
```

process.json 完整内容：

```json
{"completed":true,"seconds":77.52152163349092,"torch_peak_allocated_mib":916.21923828125}
```

complete.json：passed=true；encoder/dynamics/reward/Q/policy 的 changed_modules 均为 true；teacher_prefix_state_error=0.0；包含下表全部11次更新记录。配置 SHA256 为 `4f3c4029dbcc6e9b58362924a6929cb210ec1487aa4cdbc3665f6276678ed35e`；scope 原文为 `kernel/optimizer/interface witness; no performance claim`。

|更新|loss|consistency|reward_loss|value_loss|actor_loss|bc_loss|std_loss|grad_norm|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|2.3569579124450684|0.015197379514575005|2.6921534538269043|2.6921536922454834|1.4198219776153564|1.4198219776153564|9.475753784179688|21.903322219848633|
|2|1.7121895551681519|0.012416708283126354|2.6299686431884766|2.6397414207458496|0.8814346790313721|0.8814346790313721|5.544966220855713|16.17371940612793|
|3|1.179321527481079|0.01039750874042511|2.5492911338806152|2.5702943801879883|0.43546062707901|0.43546062707901|2.3952274322509766|2.8251116275787354|
|4|1.119599461555481|0.00919867679476738|2.4611270427703857|2.5004959106445312|0.42835181951522827|0.42835181951522827|1.1111876964569092|4.556267738342285|
|5|1.1227785348892212|0.00853774044662714|2.3715109825134277|2.4253995418548584|0.46352407336235046|0.46352407336235046|0.8808619976043701|4.009217262268066|
|6|0.9968228936195374|0.008160308934748173|2.283681869506836|2.354367971420288|0.35973140597343445|0.35973140597343445|1.0080294609069824|3.449063539505005|
|7|0.9655119180679321|0.007639596238732338|2.1922388076782227|2.2741479873657227|0.35478150844573975|0.35478150844573975|1.129981279373169|1.8709183931350708|
|8|0.9557458162307739|0.00747279217466712|2.103224515914917|2.215585231781006|0.36286506056785583|0.36286506056785583|1.154400110244751|1.5001211166381836|
|9|0.8962699770927429|0.007270880043506622|2.015024185180664|2.1438093185424805|0.3242015540599823|0.3242015540599823|1.076751470565796|1.6643214225769043|
|10|0.8485161066055298|0.007333563175052404|1.930782675743103|2.0708703994750977|0.29227739572525024|0.29227739572525024|0.940212607383728|1.6959068775177002|
|11|0.6012628078460693|0.006759271956980228|1.839421033859253|2.821352005004883|-0.42739957571029663|0.30169564485549927|0.7465837001800537|0.5919728875160217|

## 完整回合输出

```text
EPISODE_COMPLETE {"collection_index": 0, "seed": 1210000, "spawn_idx": 22, "mode": "direct_policy", "steps": 8, "ever_grasp": false, "success_once": false, "success_at_end": false, "force_limit_exceeded_any": false, "return": 2.184315502643585, "wall_seconds": 0.50993936881423, "latency_mean_ms": 0.08295103907585144, "folder": "artifacts/demo_tdmpc2_v1/witness/teacher_prefix"}
EPISODE_COMPLETE {"collection_index": 0, "seed": 1210000, "spawn_idx": 22, "mode": "direct_policy", "steps": 8, "ever_grasp": false, "success_once": false, "success_at_end": false, "force_limit_exceeded_any": false, "return": 2.199589043855667, "wall_seconds": 0.5114576108753681, "latency_mean_ms": 1.1416792403906584, "folder": "artifacts/demo_tdmpc2_v1/witness/policy_prefix"}
EPISODE_COMPLETE {"collection_index": 0, "seed": 1210000, "spawn_idx": 22, "mode": "mpc", "steps": 8, "ever_grasp": false, "success_once": false, "success_at_end": false, "force_limit_exceeded_any": false, "return": 2.1359483897686005, "wall_seconds": 1.3362793996930122, "latency_mean_ms": 104.96998205780983, "folder": "artifacts/demo_tdmpc2_v1/witness/mpc_prefix"}
```

终端另有非致命警告：functorch.combine_state_for_ensemble 弃用警告一次；ManiSkill shader_dir 弃用警告一次（2026-09-09 01:30:09,337）；mshab/envs/subtask.py:72 torch.load(weights_only=False) FutureWarning 四次。没有 traceback 或运行错误。

## 文档与实际差异、证据边界

观察到的退出码、PASS标记、77.52秒耗时、五模块更新、11条更新记录、三段各8步控制均符合文档。教师前缀状态误差为0。未发现阻断性入口差异。teacher_prefix 回合的 mode 字段也显示 direct_policy，应通过 folder 区分教师重放与学习策略，避免误标。

complete.json 未单列教师奖励误差、初始canonical配对哈希、保存加载动作误差、末步TD/MPC断言、在线混合回放比例、GPU0空闲/GPU1绑定或磁盘余量。因此本见证只能记录程序整体 passed 与可见输出，不能把这些未单列检查声称为独立审计完成；其细项需要主任务核对保存记录或实现。未读取或更改这些实现。见证不构成性能达标结论，三个8步片段均无抓取或成功。

远端证据根目录：`/data/code/ouyanghao/oyh/hab-contact-aware-world-model/artifacts/demo_tdmpc2_v1/witness`。本次未启动正式入口。
