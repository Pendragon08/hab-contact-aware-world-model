# 已有 100 回合基线归档

这是此前完成的官方 SAC 开发评估，不是本轮新的接触控制比较，也不是世界模型训练结果。

- 5 个家具布局 × 20 回合 × 200 步。
- 曾经成功 66/100，结束成功 52/100。
- 100 段完整视频、状态/动作轨迹、逐步指标、初始快照、图表和原始审计均在附件。
- 归档内 752 个文件，逐文件 SHA256 见 manifest。
- 压缩包 SHA256：`d15f3432bfec6be2c816dc766134149c910cf3f90be2f44ba9bcb671c7053757`。

`legacy-layout100` 保存正式评估；补充的 `hab-history` 保存此前 Demo、配对探针、日志、历史源码与修复记录。`pick_val_seed0_4` 的旧终端日志统计存在已记录的仪表问题，仅作历史审计；有效五回合 Demo 为 `pick_val_seed0_4_v2`，不混入 100 回合。

下载对应归档、manifest 和 release.json 后，在仓库根目录运行：

```bash
python scripts/restore_artifacts.py --inventory release/legacy-layout100.release.json --root .
python scripts/restore_artifacts.py --inventory release/hab-history.release.json --root .
```

旧基线轨迹没有保存全部逐步深度输入；本轮新的接触控制实验已增加该记录。旧快照没有完成任意中途恢复验证。
