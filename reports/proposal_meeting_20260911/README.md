# 开题方向讨论汇报材料

报告：[HTML](../../开题方向讨论汇报_世界模型动作后果排序.html) · [Markdown原稿](../开题方向讨论汇报_世界模型动作后果排序.md)。围绕研究课题、已有进展、下一阶段研究方案和方案可行性讨论四部分组织。

- `progress_snapshot.json`：北京时间2026-09-11 15:28:42只读查询的六任务进度。只统计已记录步数，不是所有任务的最终结果。
- `evaluation_evidence.json`：只读提取的Walker Walk／Pick Cube种子1、2完成记录，以及HAB已有评估与末条训练日志。去除了无关的机器资源标识；没有新增评估。
- `hab_stop_record.json`：HAB旧训练原始停止记录，时间2026-09-10 16:47左右，原因是按此前安排替换为原论文ManiSkill2任务。40万步20回合开发末步／曾成功均为0，最后回合日志426800步；不存在完整50万步完成记录。
- `input_sources.json`：本次使用的本地方案、已有报告和执行约束的SHA256。方案是待验证设计，已有原方法成功成绩不能作为新评分器效果。
- `layout.css`：报告专用离线布局；技术细节使用原生details折叠，视频不自动播放。
- `browser_validation.json`：根目录和独立副本的桌面／移动显示、链接、锚点及视频核验。

本报告基于最新证据建议先在Meta-World／ManiSkill2验证机制，将HAB接触约束作为条件扩展。没有改写2026-09-10原方案、线上Overleaf或正在运行的算法；没有启动、恢复或停止实验。

渲染：运行`scripts/build_proposal_meeting_report.py --renderer <render_html.py> --standalone-dir <仓库父目录>`。Markdown为规范原稿；其中相对链接从reports目录解析，构建器调整为HTML所在目录的路径。独立副本仍引用仓库下的本地资料与视频，移动文件时应保留目录结构。HTML review.json仅检查渲染保真，不是对科学结论的独立审稿。

本轮在线核验：TD-MPC2作者项目、BMPC arXiv及其ICLR状态、V-GPS PMLR、TD-M(PC)² L4DC2026、Sparse Imagination ICLR2026、ContactGuard／Foresight arXiv以及ManiSkill-HAB作者项目。对应直接链接均在报告中；两篇2026失败监测方法继续按预印本记录。

后续补充：第1.3节按T1—T5列出相关工作的局限与挑战，第1.4节对应C1—C3的拟贡献、方法差异和验证条件；第3.3节连接E0—E2。challenge_contribution_trace.json区分原文结论与本课题推断。第一、二项是待验证方法设计，第三项是机制与实验研究，不将普通评估包装为新控制算法。本次沿用原进度快照，没有新增实验结果。
