# 教师逐帧重建精度预检与续算

原20个拟合闭环回合已经完成，7/20曾抓取、6/20曾完整成功、0/20结束完整成功。原离线batch=8在首个教师拟合回合1210000发现动作误差0.00019749999046325684，高于冻结1e-4容差，29.93秒后停止。保留 process_offline.json 和原日志，不改变容差、不重跑闭环回合。

逐帧batch=1与原教师控制时的计算批量一致。先检查它能否满足原精度要求，再续算离线分析。环境仍d873b21f，无依赖、权重和数据划分改变，GPU1 only。

独立见证者执行以下命令一次，不修改或重试，不执行run：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 ouyanghao@10.147.17.32 "bash /data/code/ouyanghao/oyh/hab-contact-aware-world-model/scripts/run_bc_diagnosis_serial.sh witness"
```

预期exit0和BC_DIAGNOSIS_SERIAL_WITNESS_PASS。对原教师episode_000及新拟合学生episode_000的现场旁路标签，分别重建全部200个时点的batch=8和batch=1教师动作，打印每组误差。batch=1必须满足原来的1e-4；不要求batch=8通过。保存两套动作及误差数组在serial_witness，模型不更新，不对学生执行任何新动作。若失败，只汇报失败，不修复或继续。

预检通过后主任务执行 `bash scripts/run_bc_diagnosis_serial.sh run`。结果写入offline_serial，不覆盖旧offline；仍按原80条轨迹/相同统计公式计算。原学生20个留出结果不变，标签不参与训练。新预检上限240秒，与原预检47.40秒合计不超过原300秒预算；正式离线续算上限840秒，与原失败29.93秒合计不超过原900秒预算。失败时保留全部记录。
