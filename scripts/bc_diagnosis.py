"""Frozen-policy fit-start rollouts and shadow-teacher diagnostic, with no optimization."""
import argparse,json,signal,shutil,time
from pathlib import Path
import numpy as np
import torch
from bc_pipeline import ROOT,BC,ACTIVE,create_train_env,rollout,teacher_actor,load_policy,seed_all,sha
from bc_eval_v2 import canonical_pair
from legacy_evaluate_layouts import save_json

CONFIG=ROOT/'configs/bc_diagnosis_v1.json'
PLAN=json.loads(CONFIG.read_text());BASE=ROOT/'artifacts'/PLAN['version'];OLD=ROOT/'artifacts/bc_v1_2'
MOTION=[i for i in range(11) if i!=7]
def load(path):return json.loads(path.read_text())
def models(env):
    assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
    ck=torch.load(ROOT/PLAN['checkpoint'],map_location='cuda',weights_only=False)
    assert ck['update']==5000
    student=BC(ck['model']['mean'],ck['model']['std']).cuda();student.load_state_dict(ck['model']);student.eval()
    teacher,digest=load_policy(env)
    assert digest==load(ROOT/'configs/bc_v1_2.json')['checkpoint_sha256']
    return student,teacher
def selected():
    rows=load(OLD/'collection/episodes.json')
    eligible=[r for r in rows if r['split']=='fit' and r['success_at_end']]
    assert len(eligible)==53
    chosen=[eligible[i*52//19] for i in range(20)]
    assert [r['index'] for r in chosen]==PLAN['fit_collection_indices']
    return chosen
def actor(student,teacher,labels):
    teach=teacher_actor(teacher)
    def step(obs,state):
        labels.append(teach(obs,state)[0].detach().cpu().numpy().copy())
        full=torch.zeros(1,13,device='cuda');full[:,ACTIVE]=student(state);return full
    return step
def prefix_check(original,new):
    a=np.load(original/'trajectory.npz');b=np.load(new/'trajectory.npz')
    diffs={k:float(np.max(np.abs(a[k][:len(b[k])].astype(float)-b[k].astype(float)))) for k in b.files}
    assert max(diffs.values())<=PLAN['witness_prefix_tolerance'],diffs
    ea=[json.loads(x) for x in (original/'steps.jsonl').read_text().splitlines()][:8]
    eb=[json.loads(x) for x in (new/'steps.jsonl').read_text().splitlines()]
    assert ea==eb
    return diffs
def replay_teacher(teacher,folder,limit=None):
    with np.load(folder/'trajectory.npz') as archive:
        data={k:archive[k] for k in archive.files}
    initial=torch.load(folder/'initial_snapshot.pt',map_location='cpu',weights_only=False)['observation']['pixels']
    n=len(data['raw_actions13']) if limit is None else limit
    outputs=[]
    for first in range(0,n,8):
        times=range(first,min(first+8,n));pixels={}
        for k in initial:
            frames=[np.stack([initial[k][0,2+t+j].numpy() if t+j<=0 else data[k][t+j] for j in [-2,-1,0]]) for t in times]
            pixels[k]=torch.from_numpy(np.stack(frames)).float().cuda()
        state=torch.from_numpy(data['teacher_states'][first:min(first+8,n)]).float().cuda()
        with torch.no_grad():outputs.append(teacher.actor(pixels,state,compute_pi=False,compute_log_pi=False)[0].cpu().numpy())
    return np.concatenate(outputs)
def witness():
    dest=BASE/'witness';dest.mkdir(parents=True,exist_ok=False);env=create_train_env()
    try:
        student,teacher=models(env);labels=[]
        row=rollout(env,actor(student,teacher,labels),dest/'student_prefix',1210080,6,456,steps=8,spawn_idx=26)
        p=ROOT/row['folder'];original=OLD/'student/evaluation/episode_080'
        pair=canonical_pair(original,p);diffs=prefix_check(original,p)
        np.savez_compressed(dest/'shadow_actions.npz',teacher_actions13=np.stack(labels))
        replay=replay_teacher(teacher,p)
        shadow_error=float(np.max(np.abs(replay-np.stack(labels))))
        expert=replay_teacher(teacher,OLD/'collection/episode_080',limit=8)
        reference=np.load(OLD/'collection/episode_080/trajectory.npz')['raw_actions13'][:8]
        expert_error=float(np.max(np.abs(expert-reference)))
        assert max(shadow_error,expert_error)<=PLAN['teacher_replay_tolerance']
        save_json(dest/'complete.json',{'passed':True,'canonical_pair':pair,'prefix_differences':diffs,
                  'shadow_replay_max_error':shadow_error,'stored_teacher_replay_max_error':expert_error,
                  'config_sha256':sha(CONFIG),'teacher_labels_not_executed':True})
        print('BC_DIAGNOSIS_WITNESS_PASS',shadow_error,expert_error,flush=True)
    finally:env.close()
def fit():
    assert load(BASE/'witness/complete.json')['passed']
    dest=BASE/'fit';dest.mkdir(exist_ok=False);env=create_train_env();rows=[]
    try:
        student,teacher=models(env)
        for item in selected():
            labels=[];folder=dest/f"episode_{item['index']:03d}"
            row=rollout(env,actor(student,teacher,labels),folder,item['seed'],6,456,spawn_idx=item['spawn_idx'])
            save_json(folder/'initial_pair_audit.json',canonical_pair(ROOT/item['folder'],folder))
            np.savez_compressed(folder/'shadow_teacher.npz',teacher_actions13=np.stack(labels))
            row['collection_index']=item['index'];rows.append(row);save_json(dest/'episodes.json',rows)
        save_json(dest/'complete.json',{'completed':True,'episodes':20,'steps':4000,'updates':0,
                  'counts':{k:sum(r[k] for r in rows) for k in ['ever_grasp','success_once','success_at_end']}})
    finally:env.close()
def offline():
    assert load(BASE/'fit/complete.json')['completed']
    dest=BASE/'offline';dest.mkdir(exist_ok=False);env=create_train_env();rows=[]
    try:
        student,teacher=models(env)
        fitting=[r for r in load(OLD/'collection/episodes.json') if r['split']=='fit' and r['success_at_end']]
        reference=np.concatenate([np.load(ROOT/r['folder']/'trajectory.npz')['states45'][:-1,:42] for r in fitting])
        ref=(torch.from_numpy(reference).cuda()-student.mean[:42])/student.std[:42]
        groups={'teacher_fit':selected(),'teacher_dev':load(OLD/'collection/episodes.json')[80:],
                'student_fit':load(BASE/'fit/episodes.json'),'student_dev':load(OLD/'student/episodes.json')}
        for group,items in groups.items():
            for item in items:
                folder=ROOT/item['folder'];data=np.load(folder/'trajectory.npz')
                x=torch.from_numpy(data['states45'][:-1]).cuda()
                with torch.no_grad():
                    predicted=student(x).cpu().numpy()
                    normalized=(x[:,:42]-student.mean[:42])/student.std[:42]
                    distances=torch.cdist(normalized,ref,compute_mode='donot_use_mm_for_euclid_dist').amin(1).cpu().numpy()/np.sqrt(42)
                recommended=replay_teacher(teacher,folder)
                teacher_error=None
                if group.startswith('teacher'):
                    teacher_error=float(np.max(np.abs(recommended-data['raw_actions13'])))
                    assert teacher_error<=PLAN['teacher_replay_tolerance'],(group,item['seed'],teacher_error)
                else:
                    assert float(np.max(np.abs(predicted-data['effective_actions11'])))<1e-4
                    if group=='student_fit':
                        assert float(np.max(np.abs(recommended-np.load(folder/'shadow_teacher.npz')['teacher_actions13'])))<=PLAN['teacher_replay_tolerance']
                errors=(predicted-recommended[:,ACTIVE])**2
                motion=errors[:,MOTION].mean(1);gripper=errors[:,7]
                path=dest/f"{group}_{item['seed']}.npz"
                np.savez_compressed(path,student_actions11=predicted,teacher_actions13=recommended,
                     nearest_fit_state_rms=distances,motion_mse=motion,gripper_mse=gripper)
                events=[json.loads(t) for t in (folder/'steps.jsonl').read_text().splitlines()]
                rows.append({'group':group,'seed':item['seed'],'source_folder':item['folder'],
                    'analysis_file':path.relative_to(ROOT).as_posix(),'source_sha256':sha(folder/'trajectory.npz'),
                    'motion_mse':float(motion.mean()),'gripper_mse':float(gripper.mean()),'total_mse':float((motion+gripper).mean()),
                    'median_nearest_fit_rms':float(np.median(distances)),'p95_nearest_fit_rms':float(np.quantile(distances,.95)),
                    'teacher_replay_max_error':teacher_error,'ever_grasp':item['ever_grasp'],
                    'success_once':item['success_once'],'success_at_end':item['success_at_end'],
                    'first_force_limit_step':next((e['step'] for e in events if not e['cumulative_force_within_limit']),None),
                    'first_grasp_step':next((e['step'] for e in events if e['is_grasped']),None)})
                save_json(dest/'episodes.json',rows)
                print('DIAGNOSTIC_EPISODE',group,item['seed'],rows[-1]['total_mse'],flush=True)
        save_json(dest/'complete.json',{'completed':True,'episodes_analyzed':len(rows),'reference_transitions':len(reference),
                  'teacher_labels_used_for_training':False,'updates':0,'steps_advanced':0})
    finally:env.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['witness','fit','offline']);a=p.parse_args()
    BASE.mkdir(exist_ok=True)
    record=BASE/f'process_{a.stage}.json';assert not record.exists()
    assert sha(ROOT/'configs/bc_v1_2.json')==PLAN['parent_config_sha256']
    assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
    snapshot=BASE/'source_snapshot'
    if not snapshot.exists():
        snapshot.mkdir()
        for group in ['scripts','configs']:
            (snapshot/group).mkdir()
            for path in (ROOT/group).iterdir():
                if path.is_file() and path.suffix in ['.py','.sh','.json']:shutil.copyfile(path,snapshot/group/path.name)
    save_json(BASE/f'provenance_{a.stage}.json',{'source_sha256':sha(Path(__file__)),'config_sha256':sha(CONFIG),
              'checkpoint_sha256':sha(ROOT/PLAN['checkpoint']),'environment_hash':'d873b21f'})
    def timeout(sig,frame):raise TimeoutError('Frozen diagnostic stage budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(PLAN['stage_seconds'][a.stage]);start=time.time()
    try:
        seed_all(0);globals()[a.stage]()
        assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
        save_json(record,{'completed':True,'wall_seconds':time.time()-start,'peak_torch_mib':torch.cuda.max_memory_allocated()/2**20})
    except BaseException as exc:
        save_json(record,{'completed':False,'wall_seconds':time.time()-start,'error':repr(exc)});raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
