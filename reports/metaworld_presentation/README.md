# Meta-World 复现实验汇报材料

- HTML入口：`../../MetaWorld_TD-MPC2复现与实验结果汇报.html`。
- 可编辑原稿：`../MetaWorld_TD-MPC2复现与实验结果汇报.md`。
- `stage_results.csv`与`verified_results.json`来自已保存的完成证据，没有重新运行训练或新增评估。
- `learning_curves.svg/png`为标准绘图输出；HTML内嵌SVG，离线阅读无需图表网络服务。
- 正式终点8/10、阶段最好9/10；成功补录不计入正式评估。

生成顺序：安装Python绘图依赖（本次matplotlib 3.11.1），运行`scripts/build_metaworld_presentation.py`；用render-html技能的标准渲染器将原稿以academic/offline模式生成仓库根目录HTML；随后运行`scripts/package_metaworld_presentation.py --out <目标ZIP路径>`。渲染器来自本机技能安装，不属于训练环境。

便携包收录HTML直接引用的本地资料及媒体，保持相对目录结构。它不是完整训练仓库、权重包或原始数据归档。阶段原始证据入口：`hab_tdmpc2_original/reports/verified-evidence.json`。

审核记录为同系列模型的独立上下文渲染一致性检查，不构成对科研结论的独立重复实验。
