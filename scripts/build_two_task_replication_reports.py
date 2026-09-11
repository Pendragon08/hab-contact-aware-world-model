"""Build dated, evidence-based Walker Walk/Pick Cube presentation reports."""
from pathlib import Path
import argparse, base64, csv, hashlib, json, subprocess, sys
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'reports/two_task_replication_20260911'
COMMIT = '57158282b46ebc5c329c5be9cfe2b0094126d1ca'
NAMES = {
    'walker-walk': 'DMControl_Walker_Walk_TD-MPC2复现与实验结果汇报',
    'pick-cube': 'ManiSkill2_Pick_Cube_TD-MPC2复现与实验结果汇报',
}

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])

def main(renderer, selected_task=None):
    snapshot = json.loads((ASSETS/'server_snapshot.json').read_text(encoding='utf-8'))
    captured = datetime.fromisoformat(snapshot['captured_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    for key, task in snapshot['tasks'].items():
        if selected_task is not None and key != selected_task:
            continue
        walker = key == 'walker-walk'
        title = 'DMControl · Walker Walk' if walker else 'ManiSkill2 · Pick Cube'
        directory = ASSETS/key
        prefix = directory.relative_to(ROOT).as_posix()
        seeds = task['seeds']
        assert [x['seed'] for x in seeds] == [1,2,3]
        completed = [x for x in seeds if x.get('complete.json', {}).get('completed')]
        assert len(completed)==2 and not seeds[2].get('complete.json'), 'Refresh report narrative when seed3 completes.'
        metric = 'episode_reward' if walker else 'episode_success'
        metric_title = '平均回合回报（0–1000）' if walker else '末步成功率'
        scale = 1 if walker else 100
        reference = list(csv.DictReader((directory/'source'/f'{key}.csv').open(encoding='utf-8-sig')))
        ref_metric = 'reward' if walker else 'success'
        ref_final = {int(r['seed']):float(r[ref_metric]) for r in reference if int(r['step'])==1000000}
        assert set(ref_final)=={1,2,3}
        episodes = {}
        stage_rows = []
        for run in seeds:
            for row in run['evals']:
                stage_rows.append({'seed':run['seed'],'train_environment_steps':row['train_environment_steps'],
                                   'mean_return':row['episode_reward'],'terminal_success':row.get('episode_success'),
                                   'success_once_at_decision_samples':row.get('success_once')})
            if not run.get('complete.json'): continue
            c = run['complete.json']
            assert c['train_environment_steps']==1000000 and c['train_decisions']==500000
            assert c['upstream_commit']==COMMIT
            assert len(run['evals'])==11 and run['evals'][-1]['train_environment_steps']==1000000
            assert abs(run['evals'][-1][metric]-c['final_eval'][metric])<1e-6
            rows=[]
            for i in range(100,110):
                with np.load(directory/f"seed{run['seed']}"/f'episode_{i:06d}.npz',allow_pickle=False) as z:
                    assert len(z['action'])==(500 if walker else 100) and int(z['stage'])==500000
                    assert z['obs'].shape == ((501,24) if walker else (101,51))
                    assert z['action'].shape[1]==(6 if walker else 4)
                    assert all(np.isfinite(z[k]).all() for k in ['obs','action','reward','physics'])
                    rows.append({'episode':i,'return':float(z['reward'].sum()),
                                 'terminal':None if walker else bool(z['success'][-1]),
                                 'once':None if walker else bool(z['success'].any())})
            assert abs(np.mean([r['return'] for r in rows])-c['final_eval']['episode_reward'])<0.001
            if not walker:
                assert abs(np.mean([r['terminal'] for r in rows])-c['final_eval']['episode_success'])<1e-9
                assert abs(np.mean([r['once'] for r in rows])-c['final_success_once'])<1e-9
            episodes[run['seed']]=rows
        with (directory/'stage_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(stage_rows[0]));writer.writeheader();writer.writerows(stage_rows)
        dump(directory/'verified_results.json',{'captured_beijing':captured,'task':key,'completed_seeds':[1,2],
              'planned_seeds':[1,2,3],'episode_validation':episodes,'author_same_budget':ref_final,
              'source_snapshot_sha256':hashlib.sha256((ASSETS/'server_snapshot.json').read_bytes()).hexdigest()})

        fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
        colors=['#167c73','#c26429','#7465a7']
        for run,color in zip(seeds,colors):
            rows=run['evals']
            axes[0].plot([r['train_environment_steps']/1000000 for r in rows],
                         [r[metric]*scale for r in rows],marker='o',markersize=3,color=color,
                         linestyle='-' if run.get('complete.json') else ':',
                         label=f"Seed {run['seed']}"+(' (partial)' if not run.get('complete.json') else ''))
        for seed,color in zip([1,2,3],colors):
            rows=[r for r in reference if int(r['seed'])==seed and int(r['step'])<=1000000]
            axes[1].plot([int(r['step'])/1000000 for r in rows],[float(r[ref_metric])*scale for r in rows],color=color,ls='--',label=f'Author seed {seed}')
        for ax,caption in zip(axes,['Our reproduction (snapshot)','Author CSV (same budget window)']):
            ax.set(title=caption,xlabel='Training environment steps (millions)',ylabel='Episode return' if walker else 'Terminal success (%)',xlim=(0,1),ylim=(-25,1025) if walker else (-3,103))
            ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
        fig.savefig(directory/'learning_curves.png',dpi=170);plt.close(fig)

        if walker:
            fig,ax=plt.subplots(figsize=(10,3.2),constrained_layout=True)
            for seed,ep,label,color in [(1,0,'Seed1 / untrained','#a2a6ab'),(1,100,'Seed1 / 1M','#167c73'),(2,100,'Seed2 / 1M','#c26429')]:
                with np.load(directory/f'seed{seed}'/f'episode_{ep:06d}.npz',allow_pickle=False) as z:
                    ax.plot(np.arange(1,501),z['reward'],label=label,color=color,alpha=.85,lw=1)
            ax.set(xlabel='Decision index in recorded first evaluation episode',ylabel='Reward per decision (2 env steps)',ylim=(-.05,2.05))
            ax.legend();ax.grid(alpha=.2)
            detail_caption='训练前及两个最终检查点的正式评估首回合，逐决策奖励对照。每次决策的奖励是两个原生步之和；这不是完整10回合的均值，也不是成功率。'
        else:
            matrix=[]
            for ep in range(100,110):
                with np.load(directory/'seed1'/f'episode_{ep:06d}.npz',allow_pickle=False) as z:matrix.append(z['success'].astype(float))
            fig,ax=plt.subplots(figsize=(10,3.3),constrained_layout=True)
            ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=matplotlib.colors.ListedColormap(['#e6e7e9','#167c73']),vmin=0,vmax=1,extent=[.5,100.5,10.5,.5])
            ax.set(xlabel='Decision index (success observed after action repeat)',ylabel='Final evaluation episode',yticks=range(1,11),title='Pick Cube seed1: green = success; gray = not success')
            detail_caption='种子1最终10回合的成功标志。绿色为该决策结束时成功，灰色为未成功。第3回合（轨迹编号000102）曾成功但末步失败；这里只观测每次决策结束点，不涵盖动作重复中间的每个原生步。'
        fig.savefig(directory/'episode_detail.png',dpi=170);plt.close(fig)

        def fmt(value):return f'{value:.2f}' if walker else f'{round(value*10)}/10（{value:.0%}）'
        final_table=[]
        for run in seeds:
            c=run.get('complete.json');row=[str(run['seed'])]
            if c:
                row += ['已完成100万步',fmt(c['final_eval'][metric])]+([] if walker else [f"{c['final_eval']['episode_reward']:.2f}"])+[f"{c['seconds']/3600:.2f}小时"]
            else:row += [f"未完成；已记录{run['last_update']['train_environment_steps']:,}步",'暂无最终结果']+([] if walker else ['—'])+['—']
            final_table.append(row)
        stage_table=[]
        maps=[{r['train_environment_steps']:r for r in s['evals']} for s in seeds]
        for step in range(0,1000001,100000):
            stage_table.append([f'{step:,}']+[fmt(m[step][metric]) if step in m else '尚未评估' for m in maps])
        episode_table=[]
        for i in range(10):
            a,b=episodes[1][i],episodes[2][i]
            episode_table.append([str(i+1),f"{a['return']:.2f}",f"{b['return']:.2f}"]+([] if walker else ['成功' if a['terminal'] else '失败','成功' if b['terminal'] else '失败']))
        comparisons=[]
        for seed in [1,2,3]:
            c=seeds[seed-1].get('complete.json')
            comparisons.append([str(seed),fmt(c['final_eval'][metric]) if c else '未完成',f'{ref_final[seed]:.1f}' if walker else f'{ref_final[seed]:.0%}'])
        latest=seeds[2]['evals'][-1]
        robot='平面双足Walker；目标是保持身体直立并向前行走，属于运动控制，不是完整人形机器人的仓储操作。原生奖励综合身体高度、直立程度和向前速度；高回报意味着这些奖励目标完成得较好。' if walker else 'Panda机械臂抓取小方块并移到随机目标位置。它是桌面抓取定位任务，不是HAB、ManiSkill3或完整仓储搬运。目标可在空中，成功不要求把物体放到货架或松开夹爪。'
        success='主指标是原始回合累计回报：原生奖励约在0–1范围，每回合1000个原生步，因此按0–1000尺度汇报。接口中的success=0只是占位字段，不作为本任务效果指标。**980分不能写成98%成功率。**' if walker else '主指标为完整回合最后一步的原生success。已核对所用PickCube-v0源码：物体中心与目标距离≤0.025米，且机械臂非夹爪关节速度的最大绝对值≤0.2，两个条件同时满足才成功。**接近目标、抓起物体或曾经成功，都不能直接代替末步成功。**'
        metric_explanation = '''### 04.1 · 平均回合回报究竟是什么？

**它统计机器人实际执行后的环境奖励，不是奖励网络和Q网络预测出来的规划分数。**本次Walker Walk评估按以下顺序计算：

1. 冻结网络，让机器人完成一个回合：500次动作决策，对应1000个原生环境步。
2. 每个原生步由仿真环境根据身体高度、直立程度和向前运动情况给出奖励。
3. 将这个回合所有实际环境奖励相加，得到该回合的回报。
4. 对同一个模型重复评估10个完整回合，再把10个回报取平均。

<div style="padding:18px;background:#e9f2fb;border:1px solid #c7d9ed;border-radius:10px;line-height:2"><strong>单回合回报 = 该回合1000个原生环境步的实际奖励之和。</strong><br><strong>平均回合回报 = 10个回合的回报之和 ÷ 10。</strong><br>记录这个评估指标时不做折扣，也不额外加Q值。</div>

一次动作决策重复执行2个原生步，日志把这两步的实际奖励相加后保存。因此，将500条决策奖励相加，等价于将1000个原生步奖励相加。这里的“平均”是对10个完整回合取平均，不是把一个回合的总奖励再除以1000。

例如仅用两个回合演示计算：若实际回报分别为980和960，平均回合回报就是970。这两个数是教学示例，不是新增评估。**本次种子1的980.91，是其最终10个回合累计奖励的平均值；不能写成98.091%的成功率。**

### 04.2 · 它与“短期收益＋长期价值”的规划评分有什么区别？

| 对比项 | 规划时的候选评分 | 评估时的回合回报 |
|---|---|---|
| 分数从哪里来 | 奖励模型R与Q网络的预测 | 仿真环境实际返回的奖励 |
| 怎样计算 | 短期折扣预测奖励＋规划末端的折扣Q价值 | 完整回合内的实际奖励直接累加 |
| 是否加入Q估计 | 加入，补充短期窗口之后的收益 | 不加入 |
| 是否做折扣 | 本轮按γ=0.99在动作决策尺度上折扣 | 记录评估回报时不做折扣 |
| 用于什么 | 比较候选，决定下一步动作 | 衡量机器人实际完成任务的表现 |

规划评分可以概括为 `G_plan = Σ γ^t r̂_t + γ^H Q(z_H,p(z_H))`，本轮H=3。这里r̂是模型预测，不是环境已经给出的实际奖励；p(z_H)表示按策略先验产生的末端动作。评估回报则是 `G_episode = Σ r_env`，累加完整回合的真实仿真反馈。

Q会通过动作选择间接影响机器人最终获得多少奖励，但**不会作为额外的一笔分数加进评估日志**。规划分数高也不能直接证明实际回报高，必须执行后才能检验。

### 04.3 · 为什么这里不报告末步成功率？

本次使用的Walker Walk原生任务主要用连续奖励衡量“站得稳、走得好”，没有像Pick Cube那样直接用于本次评估的二元成功判据。TD-MPC2适配接口中的 `success=0` 是占位值，不能据此把所有回合判为失败。因而本报告按原方法汇报平均回合回报，而不把回报除以1000后当作成功率。

<div style="padding:16px 20px;background:#f5f1e7;border-left:4px solid #a27036;border-radius:6px"><strong>向老师汇报可以这样说</strong><p>“规划阶段用世界模型预测奖励和Q价值来选择动作；评估阶段统计机器人实际执行后获得的环境奖励。Walker Walk报告平均回合回报，Pick Cube报告末步成功率，两项任务采用不同的原生评估口径。”</p></div>

计算依据：[所用Walker奖励函数](reports/two_task_replication_20260911/walker-walk/source/native_walker.py)、[实际记录与评估入口](reports/two_task_replication_20260911/walker-walk/source/runner.py)。上面的说明解释既有结果，没有新增训练、评估或改变计分规则。

### 04.4 · 本轮评估协议与轨迹核验

''' if walker else ''
        conclusion='两个已完成种子的最终回报为980.91与977.50，说明已经形成较好的原任务行走控制表现，与作者同预算结果处于相近数值范围。尚无完整三种子统计，不据此声称优于原论文或已经完整复现论文。' if walker else '两个已完成种子的末步成功为9/10和10/10，说明世界模型＋MPC已能完成原生PickCube任务。种子1的曾成功为10/10，但末步仅9/10；种子2两者均10/10。每种子仅评估10回合，单次10/10不等于真实成功概率为100%。'
        failures='曲线已接近奖励上限，但仍有回合波动；目前没有额外评估跌倒率、抗扰动、能耗或跨地形能力。回报较高不能自动推出这些能力均达标。' if walker else '种子1最终第3回合（000102.npz）曾成功、末步失败。保存的数据可确认这一差别，但本次没有进一步拆分为“位置未达标”还是“机器人未静止”，也没有据此断言一定掉落。正式录像只记录每轮首回合，因此没有该失败回合的现成视频；保留它的原始轨迹和成功时间序列。'
        videos=[]
        for seed,name,label,ep in [(1,'eval_0000000.mp4','训练前：种子1，正式初始评估首回合',0),(1,'eval_1000000.mp4','100万步：种子1，正式最终评估首回合',100),(2,'eval_1000000.mp4','100万步：种子2，正式最终评估首回合',100)]:
            with np.load(directory/f'seed{seed}'/f'episode_{ep:06d}.npz',allow_pickle=False) as z:
                detail=f"该回合回报{z['reward'].sum():.2f}"+('' if walker else f"；末步{'成功' if z['success'][-1] else '失败'}")
            videos.append(f'<figure><figcaption><strong>{label}</strong>；{detail}。</figcaption><video controls playsinline preload="metadata" style="width:100%;max-height:430px;background:#172733" src="{prefix}/seed{seed}/{name}"></video></figure>')
        md=f'''# {title}：TD-MPC2复现与实验结果

**阶段汇报｜证据快照：北京时间 {captured}。** 本次核对服务器保存的完成标记、阶段评估日志和原始轨迹；只读取和回传资料，没有启动、暂停或重跑训练。

> 💡 {conclusion}

## 01 · 复现范围与当前结论

复现论文：Hansen、Su、Wang，**TD-MPC2: Scalable, Robust World Models for Continuous Control，ICLR 2024**。[论文PDF](https://arxiv.org/pdf/2310.16828v2) · [固定代码](https://github.com/nicklashansen/tdmpc2/tree/{COMMIT})。本任务计划3个独立训练种子，各100万原生环境步；原论文该任务训练预算为400万步。本轮是单任务、缩减预算的原方法复现，**未加入自研算法创新**。

{table(['训练种子','状态','最终'+metric_title]+([] if walker else ['最终平均回报'])+['完成记录中的用时'],final_table)}

种子3最新已保存的评估检查点为{latest['train_environment_steps']:,}环境步，{metric_title}为{fmt(latest[metric])}。它与“已记录训练步数”不是同一概念；没有100万步完成标记就不填最终成绩。以上是时间快照，阅读时训练可能继续前进。

## 02 · 仿真任务与实际配置

{robot}

{table(['设置','本次配置'],[
['任务接口','walker/walk（walker-walk）' if walker else 'PickCube-v0（pick-cube）'],
['输入与动作','24维状态、6维连续动作' if walker else '51维状态、4维连续动作；pd_ee_delta_pos'],
['仿真依赖','dm-control 1.0.8、MuJoCo 2.3.1' if walker else 'ManiSkill2 0.4.1、SAPIEN 2.2.1'],
['模型与初始化','官方约5M参数、512维潜在状态、5个Q网络；随机初始化'],
['学习数据','自行仿真交互，无教师示范、外部预训练权重或LLM API'],
['规划参数','H=3；512条候选（含24条策略候选）；64条精英；6轮搜索'],
['批量与折扣','batch=256；γ=0.99' if walker else 'batch=256；γ=0.95'],
['回合与动作重复','500次决策×2=1000原生步' if walker else '100次决策×2=200原生步'],
['每种子预算','500,000次决策×2=1,000,000原生训练步'],
['GPU','GPU0 · RTX A6000 48GB' if walker else 'GPU1 · RTX A6000 48GB']])}

模型输入为状态向量，视频用于展示；“状态观测”不等于摄像头图像输入。当前采样和控制接口保留原任务定义，没有添加新的奖励或成功条件。

## 03 · 算法怎样控制机器人

<div style="padding:18px;background:#e9f2fb;border:1px solid #c7d9ed;border-radius:10px;line-height:2"><strong>读取观察 → 编码器形成潜在状态z → 策略先验与高斯采样产生候选 → 动力学模型预测后续z → 奖励网络预测短期收益 → Q补充远期收益 → MPPI筛选并更新采样分布 → 执行首动作 → 重新观察与规划。</strong></div>

世界模型预测动作后果，策略先验提供部分候选，MPPI负责搜索。规划评分为“三步折扣预测奖励＋规划末端的折扣Q价值”。这里Q是长期回报估计，不是成功概率。候选在模型内部推演，不会逐条推进仿真；每次只执行最终选中序列的首动作。固定实现按精英权重抽取序列，评估仍有采样随机性。

训练先积累{'2500' if walker else '1000'}次决策的初始随机经验；在相应索引处集中做初始回放更新，随后通常每次决策做一次学习更新。固定原循环的随机动作判断使用“步索引≤seed_steps”，因此按索引计会多覆盖一个随机动作；不把这段初始经验学习称为专家预训练。

实际轨迹按完整回合入回放池，每批抽256个连续3步片段（4个观测、3个动作与奖励）；用潜在一致性、奖励和Q目标训练模型，再单独优化策略先验、缓慢更新目标Q。当前回合未入池时，可先用历史回合学习。这属于在线、离策略强化学习。

更详细的入门解释见[TD-MPC2算法流程汇报](TD-MPC2算法流程汇报.html)。{'例如行走时比较不同腿部动作对保持直立和向前运动的预测收益；没有人工指定固定步态。' if walker else '例如抓取时比较靠近、闭合夹爪与移动方向的预计收益；没有教师模型或人工抓取分阶段控制器。'}上述例子只解释方法，不是新的监督标签。

## 04 · 如何评估与核验

{success}

{metric_explanation}
每10万原生训练步评估一次，每轮10个完整回合；含初始和最终共11轮。冻结网络、不调用学习更新、评估经验不写入训练回放。使用MPC规划而非单独测试直接策略；固定回合长度，不因中途成功提前结束。每种子额外评估{'110,000' if walker else '22,000'}个原生步，单列于训练预算之外。

本次回传并重算种子1/2最终共20条原始轨迹：每条长度、数值有限性、观测与动作维度均检查；逐回合奖励求和与完成记录中的均值一致（允许浮点精度误差）。{'不额外生成没有原生定义的成功率。' if walker else '成功标志逐条核验得到9/10与10/10。success-once按每次动作决策结束时记录的标志计算，不涵盖动作重复期间未记录的中间原生步；末步成功则对应完整回合最后一步。'}没有为了本报告新增评估回合或筛选起点。

## 05 · 学习曲线与各阶段表现

![本次逐种子曲线与作者同预算窗口曲线]({prefix}/learning_curves.png)

左图是本次复现，右图是作者公开CSV，均为逐种子曲线，**不是两种算法的对照**。种子3仅绘制已完成评估的部分，不补齐或外推。连线连接检查点，不表示中间每一步经过评估；本报告不绘制不完整三种子的均值或置信区间。

{table(['原生训练步','种子1：'+metric_title,'种子2：'+metric_title,'种子3：阶段值'],stage_table)}

最终值与中途最好值分开：{'行走任务的高回报可能在若干阶段波动，不能挑最高检查点替代终点。' if walker else '种子1中途曾达到10/10，但100万步终点是9/10；应按终点评估如实汇报。小样本波动不能单独证明训练退化。'}原始阶段表见[CSV]({prefix}/stage_results.csv)。

## 06 · 最终10回合与失败边界

{table(['回合序号','种子1回报','种子2回报']+([] if walker else ['种子1末步','种子2末步']),episode_table)}

![正式评估轨迹细节]({prefix}/episode_detail.png)

{detail_caption}

{failures}

## 07 · 与作者同预算结果对照

{table(['种子标签','本次100万步终点','作者100万步CSV'],comparisons)}

作者参照来自固定提交中的[{key}.csv](https://github.com/nicklashansen/tdmpc2/blob/{COMMIT}/results/{key}.csv)，[本地原始副本]({prefix}/source/{key}.csv)。这里只比较100万步处的数据，不与论文400万步终点直接计算提升。相同seed数字也不保证两次实现经历了完全相同的随机轨迹，不能当作严格配对实验。{'本次已完成种子处于作者回报的相近范围。' if walker else '作者三种子在该预算为100%、80%、100%；本次已有90%、100%，缺少第三个终点，因此不比较完整三种子平均表现。'}

**可以说明：**原方法在这个任务上已学出有效控制行为。**仍不能说明：**我们提出了更好的算法、实现逐位一致复现，或已经完成原论文全部预算和统计协议。

## 08 · 正式视频：训练前与最终模型

采用预先由程序录下的“每轮第一个评估回合”，展示种子1初始、种子1终点、种子2终点，不按成功与否挑视频，不加入正式统计之外的新样本。视频均为原有训练产物，包含初始帧及每次决策后的画面，按15 FPS导出，约{'33.4' if walker else '6.7'}秒；播放时长不等于仿真物理时长。

{''.join(videos)}

## 09 · 工程改动、资源与研究局限

算法核心、网络、损失、回放和MPC沿用官方固定版本。工程适配包括精确预算终止（≤改为<）、终点显式评估、Gym动作空间单独设种子、状态与轨迹记录、{'MuJoCo离屏渲染' if walker else 'SAPIEN渲染设备绑定'}。Python3.10、PyTorch2.5.1、TorchRL/TensorDict0.6与论文历史运行时不同，不能声称完全相同的软件栈。这些不是算法创新。

用时来自每个完成记录，包含阶段评估；两种任务的物理仿真、评估长度与并发竞争不同，不能仅凭小时数断定某算法更快。当前服务器任务按每GPU三条任务流并行、每个任务种子依次运行；本报告不修改训练进程。早期README里的单进程启动限制是历史记录，不是本轮新增调度指令。

下一步先完成种子3，在相同预算、相同评估协议下汇总三种子结果；增加冻结评估回合可检验当前结果的不确定性。{'进一步可考察受扰动后的恢复或模型评分与真实回报的偏差，但这些实验尚未完成。' if walker else '进一步可拆分物体到位、机械臂静止及保持过程，诊断末步失效；再检验候选动作后果排序是否有改善空间。'}新的模块只有经过公平消融和独立评估后才能声称有效。

## 10 · 证据入口与汇报口径

- [核验结果与20条最终回合摘要]({prefix}/verified_results.json) · [带时间戳的服务器记录摘录](reports/two_task_replication_20260911/server_snapshot.json)。
- [本次启动配置]({prefix}/source/plan.json) · [运行环境规格]({prefix}/source/env-spec.json) · [只读入口源码副本]({prefix}/source/runner.py)。
- [所用原生任务源码]({prefix}/source/{'native_walker.py' if walker else 'native_pick_cube.py'}) · [原实验README快照]({prefix}/source/README.md)。这些副本供复查，不是独立可运行的完整训练环境。
- [本地资料校验清单](reports/two_task_replication_20260911/manifest.json)；报告目录保留初始及最终视频、两个已完成种子的全部最终评估轨迹和各自初始首回合轨迹。

已回传的是本次汇报所需证据子集，不含全部训练轨迹、训练权重及第三方仿真资产；不要称为全量训练数据已经备份。换电脑请保留报告旁的reports媒体目录；单独拷贝HTML可以看内嵌曲线，但不能播放缺失的视频。按Ctrl+P可打印，PDF不保留视频播放能力。

**向老师汇报可以这样说：**“我们先复现TD-MPC2的原方法，在{title}上按每种子100万环境步训练。已有两个独立种子完成，{'最终平均回报为980.91和977.50' if walker else '最终10回合末步成功分别为9次和10次'}，视频和轨迹支持实际控制效果。第三个种子尚在推进，当前是缩减预算的阶段复现结果，尚未加入自研算法改进。”
'''
        source=ROOT/'reports'/f'{NAMES[key]}.md';source.write_text(md,encoding='utf-8',newline='\n')
        output=ROOT/f'{NAMES[key]}.html'
        subprocess.run([sys.executable,str(renderer),str(source),'--out',str(output),'--offline','--title',title+'｜TD-MPC2复现与实验结果','--subtitle','100万步 · 已完成2个种子 · 阶段结果与正式视频','--eyebrow','实验汇报 · 2026-09-11'],check=True)
        html=output.read_text(encoding='utf-8')
        for name in ['learning_curves.png','episode_detail.png']:
            needle=f'src="{prefix}/{name}"';assert html.count(needle)==1
            html=html.replace(needle,'src="data:image/png;base64,'+base64.b64encode((directory/name).read_bytes()).decode()+'"')
        css = 'main img{display:block;max-width:100%;height:auto;margin:18px auto} main th,main td,main p,main a,main code{overflow-wrap:anywhere} main figure{margin:20px 0} nav.toc ol{list-style:none;padding-left:0} nav.toc ol>li::before{content:none} @media(max-width:600px){main table{display:block;max-width:100%;overflow-x:auto}}'
        html=html.replace('</head>','<style id="replication-report-layout">'+css+'</style>\n</head>',1)
        html='\n'.join(line.rstrip() for line in html.splitlines())+'\n'
        output.write_text(html,encoding='utf-8',newline='\n')
        print(f'Built {output.name}; validated 20 final evaluation episodes.')
    manifest={p.relative_to(ASSETS).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in ASSETS.rglob('*') if p.is_file() and p.name!='manifest.json'}
    dump(ASSETS/'manifest.json',manifest)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--renderer',type=Path,required=True)
    parser.add_argument('--task',choices=list(NAMES),default=None)
    args=parser.parse_args()
    main(args.renderer,args.task)
