# Walker Walk / Pick Cube 阶段复现汇报材料

快照：北京时间2026-09-11 14:27:55。两项任务种子1/2各完成100万原生环境步，种子3未完成。报告制作只读取服务器资料，不更改训练。

- [Walker Walk HTML](../../DMControl_Walker_Walk_TD-MPC2复现与实验结果汇报.html) · [可编辑原稿](../DMControl_Walker_Walk_TD-MPC2复现与实验结果汇报.md)。
- [Pick Cube HTML](../../ManiSkill2_Pick_Cube_TD-MPC2复现与实验结果汇报.html) · [可编辑原稿](../ManiSkill2_Pick_Cube_TD-MPC2复现与实验结果汇报.md)。
- server_snapshot.json保存完成记录、评估日志摘录、最后更新记录、运行时及配置；不是服务器全量训练数据备份。
- 每个任务的stage_results.csv和verified_results.json保存生成曲线的数据及20个最终回合核验结果。reward重新求和与日志误差小于0.001；Pick Cube末步success逐回合核验。
- seed1/seed2内保留全部最终10回合的NPZ以及各自初始首回合NPZ；视频为种子1初始、种子1最终和种子2最终的正式评估首回合。
- source目录为本项目入口与配置快照、作者原始结果CSV及所用原生任务源码。原实验README中的调度安排是历史记录，当前以六任务调度器为准。
- learning_curves.png和episode_detail.png是标准绘图文件，HTML内嵌图像；播放视频需要保留相对目录结构。浏览器检查见browser_validation.json；资料校验见manifest.json。

代码来源：TD-MPC2 https://github.com/nicklashansen/tdmpc2 ，固定commit57158282b46ebc5c329c5be9cfe2b0094126d1ca；DMControl1.0.8的walker.py来自 https://github.com/google-deepmind/dm_control ；ManiSkill2 0.4.1的pick_cube.py来自 https://github.com/haosulab/ManiSkill2 。第三方代码保留原项目许可与归属，本目录不包含仿真资产或第三方完整依赖。

构建入口：scripts/build_two_task_replication_reports.py --renderer <render-html技能的render_html.py绝对路径>。依赖Python、NumPy及Matplotlib；读取固定快照并核验NPZ，生成Markdown、CSV、图表及HTML。复用报告构建依赖即可，无需连接GPU或安装训练环境。新增结果应另建快照，避免覆盖历史证据；种子3完成后需更新报告范围及构建器的完成状态检查。

单独更新某份报告可追加 `--task walker-walk` 或 `--task pick-cube`；默认生成两份。Walker Walk第04章补充了实际奖励累加、10回合取平均、规划评分与评估回报的区别，以及不采用末步成功率的原因。这是既有评估口径的解释，不是新增实验结果。

汇报中的科学结论依据本次轨迹和日志核验；HTML review.json仅检查渲染一致性，不等于独立重复实验或科研结论审稿。
