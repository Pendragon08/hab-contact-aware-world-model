"""Render the measured trial, preserving active and wall-clock rates separately."""
import json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
r=json.loads((ROOT/'result.json').read_text())
samples=[json.loads(x) for x in (ROOT/'samples.jsonl').read_text().splitlines()]
start=json.loads((ROOT/'window-start.json').read_text())['wall']
end=json.loads((ROOT/'window-end.json').read_text())['wall']
ss=[s for s in samples if start<=s['wall']<=end]
hardware={}
for gpu in [0,1]:
 values=[]
 for s in ss:
  for line in s['gpu'].splitlines():
   a=[int(x.strip()) for x in line.split(',')]
   if a[0]==gpu:values.append(a)
 hardware[str(gpu)]={'mean_gpu_utilization_percent':statistics.mean(x[3] for x in values),
                     'max_used_MiB':max(x[1] for x in values),'min_free_MiB':min(x[2] for x in values),
                     'max_temperature_C':max(x[4] for x in values)}
base=sum(v['baseline_environment_steps_per_second'] for v in r['per_gpu'].values())
combined=sum(v['combined_environment_steps_per_second'] for v in r['per_gpu'].values())
summary={'measurement_seconds':r['window_seconds'],'per_gpu':r['per_gpu'],'hardware':hardware,
         'baseline_total_environment_steps_per_second':base,'concurrent_total_environment_steps_per_second':combined,
         'overall_throughput_factor':combined/base,'mean_host_load1':statistics.mean(s['load'][0] for s in ss),
         'parallel_compute_process_counts':sorted(set(len(s['compute_processes'].splitlines()) for s in ss)),
         'bounded_pilots_completed':r['completed'],'formal_jobs_continue':True,
         'limitations':['Single host, two selected tasks and one short measured window',
                       'Active rates exclude scheduled evaluation-crossing intervals; wall-window rates are in result.json',
                       'Pilot replay capacity matches formal training but its data are early-training seed2',
                       'Not a maximum-concurrency or task-quality result']}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 每张GPU两个训练进程：吞吐实测','',f"测量时间：{r['started_utc']} 至 {r['ended_utc']}（UTC），约{r['window_seconds']:.1f}秒。",'',
       '| GPU / 任务 | 原单进程 | 并发时原进程 | 并发时新增进程 | 并发合计 | 合计增益 |',
       '|---|---:|---:|---:|---:|---:|']
for tag,label in [('dmc','GPU0 / Walker Walk'),('ms2','GPU1 / Pick Cube')]:
 x=r['per_gpu'][tag]
 lines.append(f"| {label} | {x['baseline_environment_steps_per_second']:.2f} | {x['original_concurrent_rate']:.2f} | {x['pilot_concurrent_rate']:.2f} | {x['combined_environment_steps_per_second']:.2f} | {x['throughput_factor']:.2f}× |")
lines+=['',f'单位为原生环境步/秒。两张卡合计从{base:.2f}提高到{combined:.2f}，倍率{combined/base:.2f}×。',
        '', '表中为训练活跃时段速度，排除初始化、种子数据预训练，以及跨定期评估的更新区间。每个进程的真实墙钟窗口速度与剔除区间数另存result.json；不得将其混称为全流程加速。',
        '', '新增进程为同任务独立seed2短测试：5M模型、batch256、horizon3、与正式训练相同500k回放容量及仿真配置。完成测量后在完整回合处收尾、保存模型并评估。它们不计入每种子100万步的正式结果。原两个正式训练进程持续运行，冻结源码哈希已核对。',
        '', '本轮只验证这两个任务、这台机器上2进程/卡的短时吞吐，不外推为所有任务或更多并发的保证。所有原始采样、基线、日志、模型和轨迹留存在服务器throughput_trial目录。']
(ROOT/'吞吐测试结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
