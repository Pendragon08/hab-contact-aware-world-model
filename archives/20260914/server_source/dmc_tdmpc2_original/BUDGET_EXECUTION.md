# 100万步预算调整的独立检查

仿真环境spec哈希 cba67740 不变，复用已经通过的真实3000决策/2999更新验证。本次不重建环境、不改学习核心；仅缩减正式预算、调整结果目录和同预算论文对照。

按下面的命令执行一次，只报告结果，不修改、不重试、不启动训练：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=25 ouyanghao@10.147.17.32 "bash -c 'source /data/code/ouyanghao/oyh/hab-contact-aware-world-model/dmc_tdmpc2_original/runtime.sh; /data/code/ouyanghao/oyh/world-model-risk-planning/.env/bin/python /data/code/ouyanghao/oyh/hab-contact-aware-world-model/dmc_tdmpc2_original/check_budget.py'"
```

预期退出0，DMC_BUDGET_1M_CHECK_PASS。检查内容：各种子正式cfg.steps=500000决策、action repeat=2、种子1/2/3；旧runner除结果目录命名一行外逐字相同；已验证runtime/run/env-spec哈希不变；归档完成门槛与论文参照均为100万环境步。该检查不声称新的训练成绩。
