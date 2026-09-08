"""Independent raw-data audit of R001, including recovery of counts after console-log loss."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]; base=root/'artifacts/bc_v1_1/interface'
checks=['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit']
load=lambda p:json.loads(p.read_text())
rows=[];pairs=[]
for i in range(20):
    a=base/f'pair_{i:02d}/baseline';b=base/f'pair_{i:02d}/dual'
    assert load(a/'initial.json')==load(b/'initial.json')
    x=np.load(a/'trajectory.npz');y=np.load(b/'trajectory.npz')
    differences={k:float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))) for k in x.files}
    assert all(v<=1e-5 for v in differences.values())
    np.testing.assert_allclose(y['states45'][:,:42],y['teacher_states'],rtol=0,atol=1e-5)
    ea=[json.loads(s) for s in (a/'steps.jsonl').read_text().splitlines()]
    eb=[json.loads(s) for s in (b/'steps.jsonl').read_text().splitlines()]
    assert len(ea)==len(eb)==200 and ea==eb
    for folder,data,events in [(a,x,ea),(b,y,eb)]:
        row=load(folder/'episode.json'); terminal=load(folder/'terminal.json')
        assert data['teacher_states'].shape==(201,42) and data['executed_actions13'].shape==(200,13)
        assert all(np.isfinite(data[k]).all() for k in data.files)
        assert row['trajectory_sha256']==hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest()
        assert row['success_once']==any(e['success'] for e in events)==bool(terminal['s_o'])
        assert row['success_at_end']==events[-1]['success']==bool(terminal['s_e'])
        assert row['ever_grasp']==any(e['is_grasped'] for e in events)
        assert all(e['success']==all(e[k] for k in checks) for e in events)
        rows.append(row)
    pairs.append({'pair':i,'max_abs_difference':differences,'events_identical':True})
result={'passed':True,'pairs':20,'episodes':40,'steps':8000,'audits':pairs,
        'console_log_incident':'R001 console aggregate was truncated after computational completion by a lingering tee; raw per-step logs, snapshots, NPZ, videos and JSON retained. Counts reverified directly here. Console warnings cannot be fully recovered.'}
(root/'reports/bc_interface_audit.json').write_text(json.dumps(result,indent=2))
(base/'reconstructed_episode_index.json').write_text(json.dumps({'derived_from':'40 original episode.json files, not recovered console output','rows':rows},indent=2))
print('R001_RAW_AUDIT_PASS',len(rows),max(v for p in pairs for v in p['max_abs_difference'].values()))
