"""Local Chinese figures, report and offline interactive gallery."""
import csv, hashlib, html, json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from summarize_layouts import LABELS

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"results/pick_val_layout100_v1"
font_paths=[Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")]
font_path=next(p for p in font_paths if p.exists())
fontManager.addfont(str(font_path))
font=FontProperties(fname=str(font_path))
plt.rcParams.update({"font.family":["DejaVu Sans",font.get_name()],"axes.unicode_minus":False,"font.size":11})
S=json.loads((FOLDER/"summary.json").read_text(encoding="utf-8"))
rows=json.loads((FOLDER/"episodes.json").read_text(encoding="utf-8"))
layouts=json.loads((FOLDER/"by_layout.json").read_text(encoding="utf-8"))
objects=json.loads((FOLDER/"by_object.json").read_text(encoding="utf-8"))
plan=json.loads((ROOT/"configs/hab100_v1.json").read_text(encoding="utf-8"))
media=json.loads((FOLDER/"video_validation.json").read_text(encoding="utf-8"))

def pct(v): return f"{100*v:.1f}%"
def gpu_summary():
    paths=[ROOT/"logs/layout100_run_gpu.csv",ROOT/"logs/layout100_resume_gpu.csv"]
    records=[]
    for p in paths:
        if p.exists():
            with p.open(encoding="utf-8") as f:
                records.extend({k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f))
    if not records:return {}
    out={}
    for gpu in ["0","1"]:
        group=[r for r in records if r.get("index")==gpu]
        def nums(key): return [float(r[key].split()[0]) for r in group]
        if group:
            out[gpu]={"samples":len(group),"max_used_mib":max(nums("memory.used [MiB]")),
                      "min_free_mib":min(nums("memory.free [MiB]")),
                      "max_utilization_percent":max(nums("utilization.gpu [%]"))}
    return out
gpu=gpu_summary()
(FOLDER/"resource_summary.json").write_text(json.dumps(gpu,indent=2),encoding="utf-8")

x=np.arange(5); width=.33
fig,ax=plt.subplots(figsize=(10.5,5.8))
for offset,key,title,color,ci in [(-width/2,"success_once_rate","曾经成功","#4288B5","once_wilson95"),(width/2,"success_at_end_rate","结束时成功","#154B67","end_wilson95")]:
    vals=np.array([r[key] for r in layouts])*100
    bounds=np.array([r[ci] for r in layouts])*100
    ax.bar(x+offset,vals,width,label=title,color=color)
    ax.errorbar(x+offset,vals,yerr=np.maximum(0,np.array([vals-bounds[:,0],bounds[:,1]-vals])),fmt="none",color="#333333",capsize=3)
    for xx,v,r in zip(x+offset,vals,layouts):
        ax.text(xx,v+2,f"{round(v/5)}/20",ha="center",fontsize=10)
ax.set_xticks(x,[str(r["build_config_idx"]) for r in layouts])
ax.set_ylim(0,112); ax.set_ylabel("成功率（%）"); ax.set_xlabel("验证布局编号（各20回合）")
ax.set_title("官方 SAC：5 个验证布局的抓取表现",pad=14)
ax.spines[["top","right"]].set_visible(False); ax.legend(loc="upper right")
fig.text(.01,.01,"误差线为条件性 Wilson 95% 区间；单检查点、同一大场景；不代表训练种子方差。",fontsize=9,color="#555555")
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(FOLDER/"layout_success.png",dpi=180);fig.savefig(FOLDER/"layout_success.pdf");plt.close(fig)

cats=[k for k in LABELS if k!="success"]
counts=[S["outcome_counts"].get(k,0) for k in cats]
fig,ax=plt.subplots(figsize=(10.5,5.5))
bars=ax.barh([LABELS[k] for k in cats],counts,color="#B37C4C")
ax.invert_yaxis();ax.set_xlim(0,max(counts+[1])*1.23);ax.set_xlabel("回合数")
ax.set_title(f"结束未成功回合的互斥结局分类（共{100-S['success_at_end_count']}回合）",pad=14)
for b,n in zip(bars,counts):ax.text(n+.3,b.get_y()+b.get_height()/2,str(n),va="center")
ax.spines[["top","right"]].set_visible(False)
fig.text(.01,.01,"分类按预先固定的优先级；这是观测到的结局，不等于已证明的因果原因。",fontsize=9,color="#555555")
fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(FOLDER/"failure_outcomes.png",dpi=180);fig.savefig(FOLDER/"failure_outcomes.pdf");plt.close(fig)

examples=S["examples"]
fig,axes=plt.subplots(len(examples),3,figsize=(12,3.7*len(examples)),squeeze=False)
for i,example in enumerate(examples):
    extracts=[r for r in media["extracted_frames"] if r["episode_id"]==example["episode_id"]]
    for j in range(3):
        axes[i,j].axis("off")
        if j<len(extracts):
            entry=extracts[j];axes[i,j].imshow(plt.imread(FOLDER/entry["file"]))
            axes[i,j].set_title(f"回合 {entry['episode_id']} · 第 {entry['frame']} 帧",fontsize=11)
    axes[i,0].set_ylabel(LABELS[example["category"]])
fig.suptitle("各结局类别编号最小的实际回合：初始／关键／末帧",fontsize=16)
fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(FOLDER/"example_frames.png",dpi=140);plt.close(fig)

from matplotlib.colors import ListedColormap
checker_names=["抓住目标","末端回位","回收姿态","机器人稳定","接触力合规","完整成功"]
checker_keys=["is_grasped","ee_rest","robot_rest","is_static","cumulative_force_within_limit","success"]
fig,axes=plt.subplots(len(examples),1,figsize=(12,2.2*len(examples)),squeeze=False)
for i,example in enumerate(examples):
    ep_id=example["episode_id"]
    events=[json.loads(x) for x in (FOLDER/f"episode_{ep_id:03d}"/"steps.jsonl").read_text().splitlines()]
    values=np.array([[e[k] for e in events] for k in checker_keys],dtype=int)
    ax=axes[i,0]
    ax.imshow(values,aspect="auto",interpolation="nearest",cmap=ListedColormap(["#edf0f2","#26795f"]),vmin=0,vmax=1,extent=[.5,200.5,5.5,-.5])
    ax.set_yticks(range(6),checker_names,fontsize=9)
    ax.set_title(f"回合 {ep_id}：{LABELS[example['category']]}",fontsize=11,loc="left")
    ax.set_xticks([1,50,100,150,200]);ax.set_xlabel("控制步（绿色=满足，浅灰=未满足）",fontsize=9)
fig.tight_layout();fig.savefig(FOLDER/"event_timelines.png",dpi=160);plt.close(fig)

report=f"""# ManiSkill-HAB：100回合跨布局评估与失败分析

评估对象：作者公开的 SAC all-object 检查点。任务：TidyHouse / Pick 抓取子任务。此次没有训练或修改策略，没有调用 LLM API。

## 实测结论

- 曾经成功：**{S['success_once_count']}/100（{pct(S['success_once_rate'])}）**。
- 结束时成功：**{S['success_at_end_count']}/100（{pct(S['success_at_end_rate'])}）**。
- 曾经出现抓持信号：{S['ever_grasped_count']}/100；出现预定义持续失去抓持事件：{S['grasp_loss_event_episode_count']}/100。
- 至少一次累计接触力超过官方阈值：{S['force_limit_exceeded_episode_count']}/100。
- 全部100回合、20000个控制步的成功条件、终端统计和轨迹数值检查通过；100段视频均解码为201帧、512×512、20fps。

这是开发评估。5个家具布局均来自同一验证大场景sc3，不等于5套不同户型；不使用该结果宣称完整benchmark或最终测试泛化。单一官方检查点的结果不表示多次独立训练的稳定性。

原论文的子任务主要统计success once；本轮同时记录结束成功作为补充诊断。回合在首次成功后仍继续到200步，后续可能发生姿态变化或累计力超限。实际长任务技能链可能在满足条件后已切换技能，因此不能仅凭两项指标差值推断真实技能交接一定失败。

## 固定协议

从原验证集21个布局的排序索引等间隔取5个，编号69、74、79、84、89；每布局20回合。布局、种子、分类规则在预检前写入配置。原任务随机选择物体与初始化位置，没有按物体平衡，所有样本均保留。只改变布局和初始随机条件，不改动作、策略或任务成功阈值。

每回合200步；首次预检5回合、1000步，加修复后的连续重置预检10回合、2000步，共3000步单列，不计入下面100回合成绩。每次初始状态、布局、任务计划、生成索引、策略输出和逐步事件均保存。初始快照用于审计，尚未证明能精确恢复所有接触状态或完成严格配对。

首次正式运行在第22回合动作执行前发现重置回到默认布局并停止；已完成21回合布局正确。修复为每次重置明确传入固定布局，保持原环境每回合重建的行为。布局、种子、模型及评价条件均未变。续跑保留并跳过这21回合，只执行剩余79回合；前21回合文件哈希复核单独保存。错误日志、原源码、冻结配置及修复哈希记录全部保留。

策略SHA256：{S['checkpoint_sha256']}。

## 布局结果

|布局|场景文件|曾经成功|结束时成功|结束成功率95%区间|
|---|---|---|---|---|
"""
for r in layouts:
    report+=f"|{r['build_config_idx']}|{r['build_config_name']}|{r['success_once_count']}/20|{r['success_at_end_count']}/20|{pct(r['end_wilson95'][0])}–{pct(r['end_wilson95'][1])}|\n"
report+="\n![布局成绩](layout_success.png)\n\nWilson区间是给定检查点与选定布局下的描述性二项区间。相同布局内样本存在共享条件；本区间不量化训练随机性，也不保证未知场景泛化。各布局的物体与初始化构成可能不同，当前差异不能单独归因于布局难度。\n\n## 回合结局与失败条件\n\n|分类|回合数|\n|---|---|\n"
for k,label in LABELS.items():report+=f"|{label}|{S['outcome_counts'].get(k,0)}|\n"
report+="\n![失败结局](failure_outcomes.png)\n\n互斥分类优先级为：成功→全程未抓持→结束未抓持→仍抓持但力超限→仍抓持但回收姿态不满足→仍抓持且姿态达标但不稳定。它是描述结局的规则，不是因果诊断。下表保留所有失败回合末步未满足的条件，允许重复计数。\n\n|未满足条件|失败回合数（可重叠）|\n|---|---|\n"
condition_names={"is_grasped":"抓持目标","ee_rest":"末端回到目标位置","robot_rest":"机器人回收姿态","is_static":"机器人稳定","cumulative_force_within_limit":"累计接触力限制"}
for k,label in condition_names.items():report+=f"|{label}|{S['failed_episode_terminal_conditions_overlapping'].get(k,0)}|\n"
report+="\n持续失去抓持事件预定义为：至少3个连续控制步满足is_grasped，随后至少3个连续步不满足。它可能涉及放开、滑脱、碰撞或接触信号变化，不能直接称为物体掉落率。物体与末端位置、接触力和视频用于进一步判读。\n\n## 物体结果\n\n|物体|回合数|曾经成功|结束时成功|\n|---|---|---|---|\n"
for r in objects:report+=f"|{r['object']}|{r['episodes']}|{r['success_once_count']}|{r['success_at_end_count']}|\n"
report+="\n物体数量不平衡；样本少的物体只作描述，不据此给难度排名。\n\n## 视频与算力\n\n"
report+=f"各类别选择编号最小的一条回合生成[示例合集](failure_analysis_examples.mp4)，时长{media['montage_seconds']:.2f}秒，保持原始20fps、不改变播放速度。它按结局类别展示，不按总体比例抽样。全部100个原始视频在对应episode目录中。\n\n"
report+=f"平均每回合步进与录像耗时{S['mean_episode_seconds']:.2f}秒；策略网络平均推理耗时{S['policy_latency_ms']['mean']:.2f}ms、P95 {S['policy_latency_ms']['p95']:.2f}ms。后两个数只含策略前向及同步，不包含环境步进、渲染和录像，不应换算为整套系统实时控制频率。\n\n"
if gpu:
    report+=f"仅GPU1运行本项目。按秒采样的GPU1整卡占用峰值{gpu['1']['max_used_mib']:.0f}MiB（含可能的其他进程）；GPU0最小空闲{gpu['0']['min_free_mib']:.0f}MiB。采样不能保证捕获瞬时峰值。\n\n"
report+="## 后续研究决策\n\n根据本轮结局分布，先挑选具有足够重复证据的问题，再制定世界模型实验。全程未抓持较多时，优先检查目标覆盖、接近与抓取探索；抓持后失败较多时，再检查稳定收臂与接触约束。仅靠这轮失败分类不能证明世界模型或风险模块优于其他方法。\n\n本轮固定官方策略且没有算法对照；下一阶段应统一输入与数据预算建立HAB世界模型基线，风险训练使用独立训练样本，开发集用于选配置，最终测试另行保留。\n\n## 文件和来源\n\n"
report+="[逐回合数据](episodes.csv) · [逐步数据](steps.csv) · [汇总JSON](summary.json) · [视频核查](video_validation.json) · [逐回合审计](audit.json)\n\n"
report+="环境、策略、任务定义来源：[ManiSkill-HAB官方项目](https://arth-shukla.github.io/mshab/)与[官方代码](https://github.com/arth-shukla/mshab)。本报告数据全部来自本次真实仿真运行。\n"
failed_categories={k:v for k,v in S["outcome_counts"].items() if k!="success"}
if failed_categories:
    dominant=max(failed_categories,key=failed_categories.get)
    finding=f"本轮最多的失败结局为**{LABELS[dominant]}**，共{failed_categories[dominant]}回合，占结束未成功回合的{100*failed_categories[dominant]/(100-S['success_at_end_count']):.1f}%。这个占比描述当前样本，不代表因果贡献。\n\n"
    report=report.replace("## 后续研究决策\n\n","## 后续研究决策\n\n"+finding)
post_success_text="、".join(f"{LABELS[k]} {v}回合" for k,v in S["once_then_end_failure_categories"].items()) or "无"
report=report.replace("## 回合结局与失败条件\n\n",f"## 回合结局与失败条件\n\n曾经成功但结束失败共{S['once_then_end_failure_count']}回合，其中：{post_success_text}。这一统计尤其需要结合首次成功后的继续执行阶段理解。\n\n")
post_success_rows=[r for r in rows if r["success_once"] and not r["success_at_end"]]
if post_success_rows and all(r["terminal_checkers"]["is_grasped"] for r in post_success_rows):
    finding=f"**本轮{len(post_success_rows)}个曾经成功但结束失败的回合，末步均仍有抓持信号。** 因此不能把曾经成功率与结束成功率的差值直接当成掉落率。它们仍可能出现过中途接触变化，具体时序保存在事件记录中。\n\n"
    report=report.replace("## 后续研究决策\n\n","## 后续研究决策\n\n"+finding)
force_without_success=sum(r["force_limit_exceeded_any"] and not r["success_once"] for r in rows)
report=report.replace("## 后续研究决策\n\n",f"## 后续研究决策\n\n本轮{S['force_limit_exceeded_episode_count']}个回合出现过力阈值超限，其中{force_without_success}个全程未达到完整成功条件。因此接触约束问题也出现在首次成功之前，值得与接近/抓取失败分别分析。这里描述事件共现，不证明改变单一奖励或加入世界模型就能解决。\n\n")
report=report.replace("## 物体结果\n\n","## 示例事件时间线\n\n绿色表示该步满足条件，灰色表示不满足。每幅对应预先规定选例规则选中的实际回合。\n\n![事件时间线](event_timelines.png)\n\n## 物体结果\n\n")
(FOLDER/"跨布局评估报告.md").write_text(report,encoding="utf-8")

prefix="results/pick_val_layout100_v1/"
data=[]
for r in rows:data.append({**r,"category_label":LABELS[r["outcome_category"]],"video":prefix+r["video"]})
timeline=" · ".join(f"{i*10.05:.2f}s：回合{e['episode_id']} {LABELS[e['category']]}" for i,e in enumerate(examples))
layout_options="".join(f'<option value="{r["build_config_idx"]}">布局 {r["build_config_idx"]}</option>' for r in layouts)
cat_options="".join(f'<option value="{k}">{v}</option>' for k,v in LABELS.items())
payload=json.dumps(data,ensure_ascii=False).replace("</","<\\/")
page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ManiSkill-HAB · 100回合跨布局评估</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#173446;background:#f1f4f6;line-height:1.7}body{margin:0}main{max-width:1160px;margin:auto;padding:38px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:22px}.muted{color:#586c7a}.stats{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.stat{background:#173e53;color:white;padding:15px 24px;border-radius:14px}.stat b{display:block;font-size:30px}.panel{background:white;border-radius:16px;padding:24px;margin:22px 0}video{display:block;width:100%;max-height:560px;background:#152936;border-radius:10px}img{width:100%;border-radius:10px}select,button{font:inherit;padding:6px 12px;border:1px solid #c6d3db;border-radius:6px;background:white}button,a{color:#145f8c}button{cursor:pointer}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e1e8ed}.scroll{overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:800px){.grid{grid-template-columns:1fr}h1{font-size:28px}}a{overflow-wrap:anywhere}.ok{color:#237250}.fail{color:#a05a24}</style>
<main><div class="muted">MANISKILL-HAB · 官方 SAC · TidyHouse / Pick</div><h1>100 回合跨布局评估</h1>
<p class="muted">同一验证大场景的5个家具布局，每个20回合。以下是固定预训练策略的开发评估；没有训练世界模型，结果不代表完整基准或最终测试。</p>
<div class="stats"><div class="stat">实际完成<b>100 回合</b></div><div class="stat">曾经成功<b>__ONCE__ / 100</b></div><div class="stat">结束时成功<b>__END__ / 100</b></div><div class="stat">布局覆盖<b>5 个</b></div></div>
<p><a href="results/pick_val_layout100_v1/跨布局评估报告.md">完整中文报告</a> · <a href="results/pick_val_layout100_v1/episodes.csv">逐回合 CSV</a> · <a href="results/pick_val_layout100_v1/summary.json">统计记录</a></p>
<div class="grid"><section class="panel"><h2>布局成绩</h2><img src="results/pick_val_layout100_v1/layout_success.png"></section><section class="panel"><h2>失败结局</h2><img src="results/pick_val_layout100_v1/failure_outcomes.png"></section></div>
<section class="panel"><h2>示例的逐步成功条件</h2><p class="muted">绿色表示满足条件；灰色表示不满足。对应下方示例视频。</p><img src="results/pick_val_layout100_v1/event_timelines.png"></section>
<section class="panel"><h2 id="playerTitle">不同结局的真实示例合集</h2><video id="player" controls preload="metadata" src="results/pick_val_layout100_v1/failure_analysis_examples.mp4"></video><p id="description" class="muted">__TIMELINE__</p><p>示例按各类别最小回合编号选取，保留成功和失败，不代表类别频率；视频20fps播放不等于计算实时速度。</p></section>
<section class="panel"><h2>全部100回合</h2><p><select id="layout"><option value="">全部布局</option>__LAYOUTS__</select> <select id="category"><option value="">全部结局</option>__CATEGORIES__</select> <span id="count"></span></p>
<div class="scroll"><table><thead><tr><th>回合</th><th>布局</th><th>目标物体</th><th>曾经成功</th><th>结束成功</th><th>观测结局</th><th>视频</th></tr></thead><tbody id="body"></tbody></table></div></section>
<p class="muted">“持续失去抓持”是预定义接触信号事件，不能直接等同于物体掉落。分类描述观测结局，不能独立证明失败的因果机制。<br>来源：<a href="https://arth-shukla.github.io/mshab/">ManiSkill-HAB 作者环境与公开策略</a>。</p></main>
<script>const rows=__DATA__;const body=document.getElementById('body');function render(){body.replaceChildren();const selected=rows.filter(r=>(!layout.value||String(r.build_config_idx)===layout.value)&&(!category.value||r.outcome_category===category.value));document.getElementById('count').textContent=selected.length+' 条';for(const r of selected){const tr=document.createElement('tr');for(const x of [r.episode_id,r.build_config_idx,r.target_object,r.success_once?'是':'否',r.success_at_end?'是':'否',r.category_label]){const td=document.createElement('td');td.textContent=x;tr.append(td)}const td=document.createElement('td');const b=document.createElement('button');b.textContent='播放';b.onclick=()=>{const player=document.getElementById('player');player.src=r.video;document.getElementById('playerTitle').textContent='回合 '+r.episode_id+' · '+r.category_label;document.getElementById('description').textContent='布局 '+r.build_config_idx+' · '+r.target_object+' · 种子 '+r.seed;player.scrollIntoView({behavior:'smooth',block:'center'});player.play().catch(()=>{})};td.append(b);tr.append(td);body.append(tr)}}const layout=document.getElementById('layout'),category=document.getElementById('category');layout.onchange=category.onchange=render;render();</script></html>"""
page=page.replace("__ONCE__",str(S["success_once_count"])).replace("__END__",str(S["success_at_end_count"])).replace("__TIMELINE__",html.escape(timeline)).replace("__LAYOUTS__",layout_options).replace("__CATEGORIES__",cat_options).replace("__DATA__",payload)
(ROOT/"跨布局评估汇总.html").write_text(page,encoding="utf-8")
files=[ROOT/"跨布局评估汇总.html",ROOT/"configs/hab100_v1.json",ROOT/"configs/hab100_v1_repair.json",ROOT/"logs/layout100_run_v1.log",ROOT/"logs/layout100_run_gpu.csv",ROOT/"logs/layout100_resume_v1.log",ROOT/"logs/layout100_resume_gpu.csv"]+[ROOT/"scripts"/name for name in ["evaluate_layouts.py","summarize_layouts.py","validate_layout_media.py","build_layout_report.py"]]+sorted(p for p in FOLDER.rglob("*") if p.is_file())
manifest={"task":"HAB Pick cross-layout development evaluation","files":[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
(ROOT/"layout100_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print("REPORT_READY",FOLDER/"跨布局评估报告.md")
