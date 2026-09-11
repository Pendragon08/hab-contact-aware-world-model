# 高水平文献与可执行课题 · 2026-09-11

26篇正式2024—2026论文（6篇Q1期刊、20篇顶会），每篇记录研究问题、贡献、方法、任务、指标、局限、代码、数据和Zotero入口。8篇新增、18篇复用，26篇PDF已在本地Zotero核验。

- 主报告源稿：上级目录 `世界模型与机器人操作_高水平文献与可执行课题报告.md`
- `papers.json`：26篇完整元数据和阅读卡；`world_model_topics_26refs.bib`：Overleaf可用引用。
- `文献与Zotero核验清单.csv`、`zotero_audit.json`：逐条入库与附件核验；`jcr_evidence.txt`：2024 JCR资格来源。
- 根目录 `scripts/build_world_model_topic_report.py` 生成HTML；完整文献PDF保存在Zotero，不包含在Git仓库。

未开展新训练，未编辑Overleaf，未确认Zotero云同步。候选贡献仍待验证，JIF明确使用2024年度，不称为2026最新。

## 本轮深化：逐篇局限 → 三项挑战 → 对应方法

五个课题现在分别包含逐篇文献局限表与三个挑战／方法，共41条引用定位，覆盖同一26篇论文。每项方案包含输入、训练信号、必要对照、指标和否定条件；新增第05—11节是汇报主线。FLAIR既有异常过滤、WIMLE置信度视界未来方向、SimDist冻结价值上限等先行工作已明确标出。证据见 `evidence_topic_challenges/topic_evidence.json`，阅读卡的相关局限也已同步修订。

原生任务修订：第01节主选RoboCasa PickPlaceCounterToCabinet，列出三项任务内挑战、官方数据与评估入口；既有StackCube作独立起点，MS-HAB官方完整任务作备选。撤下自设仓储工位、物理扰动和遮挡变体；A/D保留为停用的文献候选，B/C/E按原生协议验证。未启动新实验。
