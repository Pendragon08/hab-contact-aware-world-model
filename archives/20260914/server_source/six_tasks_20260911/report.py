"""Render the observed launch and throughput evidence, without performance claims."""
import datetime,json,pathlib,shutil,subprocess
from sample import snapshot
ROOT=pathlib.Path(__file__).resolve().parent
def bj(epoch):return datetime.datetime.fromtimestamp(epoch,datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
r=json.loads((ROOT/'throughput-comparison.json').read_text());s=snapshot();state=json.loads((ROOT/'state.json').read_text());s['manager_state']=state
(ROOT/'latest-status.json').write_text(json.dumps(s,indent=2)+'\n')
thermal=subprocess.check_output(['nvidia-smi','-q','-d','TEMPERATURE,PERFORMANCE'],text=True)
(ROOT/'thermal-status.txt').write_text(thermal)
lines=['# 六任务启动与吞吐实测', '',f'状态检查时间：{bj(s["time"])}（北京时间）。正式提交时间：{bj(json.loads((ROOT/"launch.json").read_text())["utc_epoch"])}。','',
 'GPU0：Walker Walk、Walker Run、新增 Cheetah Run。GPU1：Pick Cube、Lift Cube、新增 Stack Cube。新增任务各3种子，每种子100万原生训练环境步；评估交互另计。正式种子之间顺序执行，任务之间并发。','',
 '原四个训练进程在交接时保留 PID 和进度，未停止或重启。后续正常换种子由新调度器执行，当前统一状态以 six_tasks_20260911/state.json 为准。旧调度器已退役，不应手动恢复。','',
 '| 任务 | GPU | 当前种子 | 已记录训练环境步 | 参数更新 | PID | 状态 |','|---|---:|---:|---:|---:|---:|---|']
for task,st in state['streams'].items():
 data=s['tasks'][task];seed=next(x for x in data['seeds'] if x['seed']==st['seed'])
 lines.append(f'| {task} | {data["gpu"]} | {st["seed"]} | {seed["steps"]:,} | {seed["updates"]:,} | {st.get("pid")} | {st["status"]} |')
lines+=['','## 吞吐比较','','新增前后各采集约五分钟墙钟窗口，六任务窗口在全部任务通过初始种子数据预训练后开始。','',
 '| 指标 | 四任务 | 六任务 | 增幅 |','|---|---:|---:|---:|']
for g in ['0','1']:
 lines.append(f'| GPU{g} 合计环境步/秒 | {r["four"]["gpu"][g]["steps_per_second"]:.2f} | {r["six"]["gpu"][g]["steps_per_second"]:.2f} | {(r["gpu_ratios"][g]-1)*100:.1f}% |')
lines.append(f'| 两卡合计环境步/秒 | {r["four"]["total_steps_per_second"]:.2f} | {r["six"]["total_steps_per_second"]:.2f} | {(r["aggregate_ratio"]-1)*100:.1f}% |')
lines+=['','| GPU | 六任务平均利用率 | 窗口峰值显存 | 最高温度 |','|---|---:|---:|---:|']
for g in ['0','1']:
 v=r['six']['gpu'][g];lines.append(f'| {g} | {v["mean_utilization"]:.1f}% | {v["peak_memory_MiB"]/1024:.2f} GiB | {v["max_temperature_C"]:.0f}°C |')
lines+=['',f'四任务采样：{r["four"]["start_utc"]} 至 {r["four"]["end_utc"]}。六任务采样：{r["six"]["start_utc"]} 至 {r["six"]["end_utc"]}（UTC）。',
 f'六任务窗口观察到的 CUDA 计算进程数：{r["six"]["compute_process_counts"]}。',
 '这是不同任务组合、可能不同种子阶段的一次运行吞吐比较；包括评估和记录开销，步数日志有200环境步量化。总吞吐增加不表示每个任务更快，也不表示学习效果改善或并发上限已确定。短验证期间驱动记录温控降频；最新驱动温控状态保存在 thermal-status.txt。显存余量不能代表计算或散热余量。','',
 '## 验证与留存','',
 '独立执行者 /root/six_task_witness 按文档执行一次并通过：Cheetah6000训练环境步/2999次更新，Stack2400步/1199次更新；两者全部五模块改变、日志有限、轨迹检查误差0。Stack原生成功正例/反例均正确，GPU1承担CUDA计算，GPU0仅4MiB图形枚举上下文。短验证不是正式性能结果。',
 '独立验证指出的差异是根代理额外暂缓了旧调度器的换种子，记录在 validation-scheduler-hold.json；所有训练子进程均保留。调度替换另有四个临时CPU子进程的集成测试，通过后才正式交接。',
 '保留原任务源文件与上游算法哈希。新任务仍是缩短至100万步的官方方案复现，历史Python/PyTorch/TorchRL运行时差异沿用此前披露。完成每个正式种子后自动生成服务器归档和校验清单；这不等于所有实验已经完成。',
 f'本次检查数据盘可用：{shutil.disk_usage(ROOT).free/2**30:.2f} GiB，运行保护线15 GiB。',
 '新增任务日志：cheetah_run/logs/seed1_1m.log、stack_cube/logs/seed1_1m.log。全部任务的只读状态入口：status.py。原始速度采样：measurements；结构化比较：throughput-comparison.json。','']
(ROOT/'launch-report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(checked_beijing=bj(s['time']),ratio=r['aggregate_ratio'],gpu=s['gpu'],compute=s['compute'],disk_free_GiB=shutil.disk_usage(ROOT).free/2**30),indent=2))
