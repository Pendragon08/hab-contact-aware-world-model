"""Compact read-only snapshot of BC stages; no simulator import or GPU allocation."""
import json, os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
plan=json.loads((root/os.environ.get('HAB_BC_CONFIG','configs/bc_v1.json')).read_text()); base=root/'artifacts'/plan['version']
result={'version':plan['version']}
for stage in ['witness','interface','collect','train']:
    path=base/f'process_{stage}.json'
    if path.exists(): result[stage]=json.loads(path.read_text())
for label,path in [('interface_pairs',base/'interface/progress.json'),('teacher_episodes',base/'collection/episodes.json'),('student_episodes',base/'student/episodes.json')]:
    if not path.exists():continue
    data=json.loads(path.read_text()); rows=data['rows'] if isinstance(data,dict) else data
    result[label]={'completed':len(rows),'ever_grasp':sum(r['ever_grasp'] for r in rows),
                   'success_once':sum(r['success_once'] for r in rows),'success_at_end':sum(r['success_at_end'] for r in rows)}
    if isinstance(data,dict):result[label]['paired_audits_passed']=sum(r['passed'] for r in data['audits'])
path=base/'student/metrics.json'
if path.exists():result['last_training_metric']=json.loads(path.read_text())[-1]
path=base/'student/complete.json'
if path.exists():result['student_result']=json.loads(path.read_text())
print(json.dumps(result,ensure_ascii=False))
