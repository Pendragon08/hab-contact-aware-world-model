"""Auxiliary absolute-velocity diagnostic; never changes frozen HAB metrics or fit filtering."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
checks=['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit']
paths=list((root/'artifacts/bc_v1_1/interface').glob('pair_*/baseline/episode.json'))
paths+=list((root/'artifacts/bc_v1_2/collection').glob('episode_*/episode.json'))
paths+=list((root/'artifacts/bc_v1_2/student/evaluation').glob('episode_*/episode.json'))
rows=[]
for path in sorted(paths):
    row=json.loads(path.read_text()); folder=path.parent
    d=np.load(folder/'trajectory.npz'); state=d['physical_states']; assert state.shape==(201,44)
    qvel=state[1:,15:30]
    original=(qvel[:,3:-2]<=.2).all(1)&(qvel[:,:3]<=.05).all(1)
    absolute=(np.abs(qvel[:,3:-2])<=.2).all(1)&(np.abs(qvel[:,:3])<=.05).all(1)
    events=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    np.testing.assert_array_equal(original,[e['is_static'] for e in events])
    original_success=np.array([e['success'] for e in events],dtype=bool)
    strict_success=np.array([all(e[k] for k in checks if k!='is_static') for e in events])&absolute
    if '/interface/' in path.as_posix():group='interface_teacher'
    elif '/student/' in path.as_posix():group='student_dev'
    else:group='teacher_fit_all' if int(folder.name.split('_')[-1])<80 else 'teacher_dev'
    rows.append({'group':group,'seed':row['seed'],'folder':folder.relative_to(root).as_posix(),
                 'static_boolean_difference_steps':int(np.sum(original!=absolute)),
                 'original_success_once':bool(original_success.any()),'original_success_end':bool(original_success[-1]),
                 'absolute_speed_success_once':bool(strict_success.any()),'absolute_speed_success_end':bool(strict_success[-1])})
groups={}
for group in sorted({r['group'] for r in rows}):
    part=[r for r in rows if r['group']==group]
    groups[group]={'episodes':len(part),**{k:sum(r[k] for r in part) for k in ['static_boolean_difference_steps','original_success_once','original_success_end','absolute_speed_success_once','absolute_speed_success_end']}}
result={'passed':True,'scope':'auxiliary diagnostic only; original HAB success, training filter and gate are unchanged',
        'source_commit':'17121e3f96e3ee3ed0c03610b17f8bc2864617af',
        'source':'mani_skill/agents/robots/fetch/fetch.py:400-405; called by mshab sequential_task.py',
        'finding':'Pinned upstream is_static compares signed joint velocities to positive thresholds, without abs. Auxiliary diagnostic applies abs to the same body/base thresholds.',
        'observation_limit':'HAB native state removes first three base qpos/qvel entries. 45D BC has no explicit base velocity or image history; teacher additionally has three depth frames.',
        'groups':groups,'rows':rows}
(root/'reports/bc_semantics_audit.json').write_text(json.dumps(result,indent=2))
print('BC_SEMANTICS_DIAGNOSTIC',json.dumps(groups))
