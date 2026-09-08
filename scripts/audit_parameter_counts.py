"""Correct report accounting: launch SAC hash/count covered its actor only."""
import json,os
from pathlib import Path
import torch
root=Path(os.environ['HAB_ROOT']);path=root/'reports/state_pilot/summary.json'
summaries=json.loads(path.read_text());audit={}
for algo,row in summaries.items():
    ckpt=torch.load(root/f'artifacts/state_{algo}_pilot_v1/policy.pt',map_location='cpu',weights_only=False)
    if algo=='sac':
        counts={k:sum(v.numel() for v in ckpt[k].values() if isinstance(v,torch.Tensor)) for k in ['actor','q1','q2']}
        counts['entropy_temperature']=1
        total=sum(counts.values());scope='actor only'
    else:
        counts={'model_without_target_Q':sum(v.numel() for k,v in ckpt['model'].items() if not k.startswith('_target_Qs.') and isinstance(v,torch.Tensor))}
        total=sum(counts.values());scope='model trainable parameters, excluding target Q'
    original=json.loads((root/f'artifacts/state_{algo}_pilot_v1/provenance.json').read_text())['trainable_model_parameters']
    audit[algo]={'launch_recorded_count':original,'launch_count_scope':scope,'components':counts,'total_trainable_parameters':total}
    row['launch_parameter_subset_count']=original;row['launch_parameter_subset_scope']=scope
    row['trainable_model_parameters']=total
path.write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
(path.parent/'parameter_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
report=path.parent/'报告.md'
if '参数计数补充：' not in report.read_text(encoding='utf-8'):
    with report.open('a',encoding='utf-8') as f:
        f.write('\n参数计数补充：原始 SAC 启动记录只对 actor 做参数变化哈希和计数。汇总表的模型参数量已根据保存的 actor、双 Q 和熵温度重新计算；原始启动记录保持不变。TD-MPC2 排除目标 Q 后计数。详细分项见 `parameter_audit.json`。\n')
print('PARAMETER_ACCOUNTING_AUDIT',json.dumps(audit),flush=True)
