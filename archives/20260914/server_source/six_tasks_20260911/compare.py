"""Compare observed aggregate throughput with explicit sampling-window limitations."""
import json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
def summarize(label):
 rows=[json.loads(x) for x in (ROOT/'measurements'/f'{label}.jsonl').read_text().splitlines()]
 a,b=rows[0],rows[-1];dt=b['time']-a['time'];assert dt>=300
 tasks={}
 for name,data in b['tasks'].items():
  delta=data['total_steps']-a['tasks'].get(name,{}).get('total_steps',0)
  tasks[name]=dict(gpu=data['gpu'],steps_per_second=delta/dt,delta_steps=delta)
 gpu={}
 for index in [0,1]:
  samples=[]
  for r in rows:
   samples.append(next([float(x.strip()) for x in line.split(',')] for line in r['gpu'].splitlines() if int(line.split(',')[0])==index))
  gpu[str(index)]=dict(steps_per_second=sum(v['steps_per_second'] for v in tasks.values() if v['gpu']==index),mean_utilization=statistics.mean(v[3] for v in samples),peak_memory_MiB=max(v[1] for v in samples),max_temperature_C=max(v[4] for v in samples))
 return dict(start_utc=a['utc'],end_utc=b['utc'],seconds=dt,samples=len(rows),tasks=tasks,gpu=gpu,total_steps_per_second=sum(v['steps_per_second'] for v in tasks.values()),compute_process_counts=sorted(set(len(r['compute'].splitlines()) for r in rows)))
if __name__=='__main__':
 before=summarize('four_baseline');after=summarize('six_steady')
 report=dict(four=before,six=after,aggregate_ratio=after['total_steps_per_second']/before['total_steps_per_second'],gpu_ratios={g:after['gpu'][g]['steps_per_second']/before['gpu'][g]['steps_per_second'] for g in ['0','1']},limitations='One5minute window per condition; different task mixture and possibly seed phase. Logged steps have200native-step quantization. Includes evaluations/recording overhead. Aggregate environment-step rate is operational throughput, not learning efficiency or a guaranteed maximum concurrency.')
 (ROOT/'throughput-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
