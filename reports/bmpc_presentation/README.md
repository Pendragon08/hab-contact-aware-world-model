# BMPC 汇报的原图、数据与核验

[报告原稿](../BMPC论文汇报.md) · [离线 HTML](../../BMPC论文汇报.html)

论文为 Wang 等人的 *Bootstrapped Model Predictive Control*，ICLR 2025。本材料来自用户已有的 arXiv:2503.18871v2 PDF（19页）、正式会议页面及固定版官方代码；没有执行 BMPC 训练。

| 文件 | 内容与来源 |
|---|---|
| `figure1.png` | PDF第2页原图1：整体算法与策略比较 |
| `algorithm1.png` | PDF第6页算法1 |
| `figure4.png` | PDF第8页原图4：7项高维DMC曲线 |
| `figure7.png` | PDF第10页原图7：算法变体及重分析消融 |
| `tasks.png` | PDF第19页图11/12：两类平台任务 |
| `official_data_dmc_hard_4M.json` | 官方固定提交的公开曲线原文件 |
| `selected_official_results.csv` | 7项任务在1M/4M步时的两算法均值，直接取JSON字段 |
| `BMPC_seven_task_mean.csv`、`tdmpc2_seven_task_mean.csv` | 相同原生步数处7个任务的等权平均 |
| `evidence.json` | PDF/图片/数据SHA256、裁图页码坐标、固定提交、阈值计算及差异说明 |
| `browser_validation.json` | 页面结构、内嵌图片、目录锚点、离线和两种屏宽检查 |
| `presentation.css` | 内嵌到HTML的响应式布局补充 |

图片是原论文局部摘录，保留原图与图注，版权归作者；不是重新生成的实验图。算法1用原始公式显示，其余讲解使用离线可读的文本公式。HTML内嵌五个PNG文件，无需CDN。

官方提交：`d5dabc6cb58d463829571548c47711dff074f5af`；[原始数据](https://github.com/wertyuilife2/bmpc/blob/d5dabc6cb58d463829571548c47711dff074f5af/results/data_dmc_hard_4M.json)。曲线的`values`是作者发布的均值，不能据此恢复所有种子的原始轨迹。本报告不重新估计置信区间。

阈值核验将dog-stand/trot/walk/run、humanoid-stand/walk/run这7项在相同步数的均值等权平均，寻找首次记录值≥795，不插值。结果为900,000和3,600,000步；正文写90k/360k，保留差异，未获作者确认。

重新生成HTML，在仓库根目录执行：

```bash
python scripts/build_bmpc_presentation.py --renderer /path/to/render-html/scripts/render_html.py
```

修改原稿后需重新生成并检查，不直接手改HTML。图片哈希由构建脚本校验。渲染审阅只验证内容保真与结构，不代表独立复核所有科学结论。
