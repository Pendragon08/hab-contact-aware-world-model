# TD-MPC2论文原图来源

`figure3.png`为论文Figure 3的图面裁切，未重绘。

- 作者：Nicklas Hansen、Hao Su、Xiaolong Wang。
- 论文：TD-MPC2: Scalable, Robust World Models for Continuous Control，ICLR 2024。
- 版本：[arXiv:2310.16828v2](https://arxiv.org/pdf/2310.16828v2)，2024-03-21，第3页。
- 原作许可：CC BY 4.0，见[arXiv网页版许可信息](https://arxiv.org/html/2310.16828v2)。
- 来源：用户已下载的本地PDF；PyMuPDF按PDF点坐标(336, 412, 505, 551)裁切，以4倍分辨率栅格化，仅去除图外正文和原图注。报告提供中文图注与阅读说明。
- 使用`python scripts/build_tdmpc2_presentation.py --renderer <render-html技能的render_html.py绝对路径>`从Markdown重新生成报告。生成器只内嵌这一固定PNG，确保报告根目录副本和便携包无须额外图片文件也可离线显示。
- 报告内容依据论文与固定版官方实现整理；图中所有网络设计归原作者，中文说明不代表本项目的新算法成果。

## 第06—08章汇报编排（2026-09-11）

第06章讲控制与学习两个过程，第07章按信息流解释五个模块并对照原图，第08章走完一次动作决策。每章都有口述稿，建议主线约1＋2＋2分钟。

原有论文图逐项说明和MPPI公式、参数、数值例子保留在四个原生details展开区中；点击summary可展开。主章节标题与原链接保持，原08.x补充标题仍有锚点。打印补充内容前应先展开对应区域。

样式来自talk-layout.css，由构建脚本内嵌，无需联网或额外JavaScript。talk-browser-validation.json记录桌面/手机、收起/全部展开、锚点及重复ID检查。

## 规划与训练相邻（2026-09-11）

章节顺序调整为08规划、09训练、10局限性，训练小节改为09.1—09.6；正文中的训练章节引用同步更新。八个旧主／小节锚点作为别名保留，已有书签仍能找到原内容。training-order-validation.json记录17章正文保留核验、目录顺序、旧链接、桌面／手机显示和根目录副本一致性。
