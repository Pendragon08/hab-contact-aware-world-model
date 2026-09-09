"""Read-only progress and measured ETA for the two original-core baselines."""
import pathlib,json,datetime,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1]

def inspect_run(folder,repeat,budget,historical_seconds=None):
    p=ROOT/folder/'artifacts/seed1_500k';f=p/'metrics.jsonl'
    result={'folder':folder,'budget_native_steps':budget,'action_repeat':repeat,'complete':(p/'complete.json').exists()}
    if not f.exists():
        result.update(status='not_started',historical_training_seconds_estimate=historical_seconds)
        return result
    rows=[]
    for line in f.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass # Ignore only an incomplete last write.
    train=[x for x in rows if x['category']=='train' and x['step']>=1200]
    if not train:result['status']='initial_evaluation_or_warmup';return result
    last=train[-1];window=train[-100:];first=window[0]
    result.update(status='complete' if result['complete'] else 'training',native_steps=last['step']*repeat,
                  latest_total_time=last['total_time'],latest_loss=last.get('total_loss'))
    if last['step']>first['step']:
        speed=(last['step']-first['step'])*repeat/(last['total_time']-first['total_time'])
        result.update(recent_native_steps_per_second=speed,eta_training_seconds=max(0,budget-last['step']*repeat)/speed,
                      eta_window_native_steps=(last['step']-first['step'])*repeat,
                      eta_note='Recent measured throughput; future evaluation/archive and host-load changes may add time.')
    evals=[x for x in rows if x['category']=='eval']
    if evals:result['latest_evaluation']=evals[-1]
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out');args=p.parse_args()
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'metaworld':inspect_run('tdmpc2_original',2,500000),
            'hab':inspect_run('hab_tdmpc2_original',1,500000,2978.120866684243*50)}
    q=ROOT/'hab_tdmpc2_original/artifacts/queue/status.json'
    if q.exists():result['hab_queue']=json.loads(q.read_text())
    if args.out:
        target=pathlib.Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
