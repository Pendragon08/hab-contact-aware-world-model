"""Serial teacher replay recovery; preserve the batch-eight failure and original tolerance."""
import argparse,json,signal,time,shutil
from pathlib import Path
import numpy as np
import torch
from bc_diagnosis import ROOT,BASE,OLD,PLAN,CONFIG,MOTION,load,models,selected,replay_teacher
from bc_pipeline import ACTIVE,create_train_env,seed_all,sha
from legacy_evaluate_layouts import save_json

def serial_teacher(teacher,folder):
    with np.load(folder/'trajectory.npz') as archive:data={k:archive[k] for k in archive.files}
    initial=torch.load(folder/'initial_snapshot.pt',map_location='cpu',weights_only=False)['observation']['pixels']
    outputs=[]
    for t in range(len(data['raw_actions13'])):
        pixels={k:torch.from_numpy(np.stack([initial[k][0,2+t+j].numpy() if t+j<=0 else data[k][t+j] for j in [-2,-1,0]]))[None].float().cuda() for k in initial}
        state=torch.from_numpy(data['teacher_states'][t:t+1]).float().cuda()
        with torch.no_grad():outputs.append(teacher.actor(pixels,state,compute_pi=False,compute_log_pi=False)[0][0].cpu().numpy())
    return np.stack(outputs)

def witness():
    dest=BASE/'serial_witness';dest.mkdir(exist_ok=False);env=create_train_env();rows=[]
    try:
        student,teacher=models(env)
        for label,folder in [('original_teacher',OLD/'collection/episode_000'),('live_shadow',BASE/'fit/episode_000')]:
            batched=replay_teacher(teacher,folder);serial=serial_teacher(teacher,folder)
            data=np.load(folder/('trajectory.npz' if label=='original_teacher' else 'shadow_teacher.npz'))
            reference=data['raw_actions13' if label=='original_teacher' else 'teacher_actions13']
            e8=np.abs(batched-reference).max(1);e1=np.abs(serial-reference).max(1)
            assert float(e1.max())<=PLAN['teacher_replay_tolerance']
            np.savez_compressed(dest/f'{label}.npz',batch8=batched,batch1=serial,reference=reference,batch8_error=e8,batch1_error=e1)
            rows.append({'source':label,'folder':folder.relative_to(ROOT).as_posix(),'batch8_max_error':float(e8.max()),
                         'serial_max_error':float(e1.max()),'batch8_worst_step_before_action':int(e8.argmax())})
        save_json(dest/'complete.json',{'passed':True,'rows':rows,'threshold_unchanged':PLAN['teacher_replay_tolerance'],
                  'simulation_control_steps':0,'updates':0,'config_sha256':sha(CONFIG)})
        print('BC_DIAGNOSIS_SERIAL_WITNESS_PASS',rows,flush=True)
    finally:env.close()

def run():
    assert load(BASE/'serial_witness/complete.json')['passed']
    assert load(BASE/'fit/complete.json')['completed']
    dest=BASE/'offline_serial';dest.mkdir(exist_ok=False);env=create_train_env();rows=[]
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
                recommended=serial_teacher(teacher,folder);teacher_error=None
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
                events=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
                rows.append({'group':group,'seed':item['seed'],'source_folder':item['folder'],
                    'analysis_file':path.relative_to(ROOT).as_posix(),'source_sha256':sha(folder/'trajectory.npz'),
                    'motion_mse':float(motion.mean()),'gripper_mse':float(gripper.mean()),'total_mse':float((motion+gripper).mean()),
                    'median_nearest_fit_rms':float(np.median(distances)),'p95_nearest_fit_rms':float(np.quantile(distances,.95)),
                    'teacher_replay_max_error':teacher_error,'ever_grasp':item['ever_grasp'],
                    'success_once':item['success_once'],'success_at_end':item['success_at_end'],
                    'first_force_limit_step':next((e['step'] for e in events if not e['cumulative_force_within_limit']),None),
                    'first_grasp_step':next((e['step'] for e in events if e['is_grasped']),None)})
                save_json(dest/'episodes.json',rows);print('DIAGNOSTIC_SERIAL_EPISODE',group,item['seed'],rows[-1]['total_mse'],flush=True)
        save_json(dest/'complete.json',{'completed':True,'episodes_analyzed':80,'reference_transitions':len(reference),
                  'teacher_labels_used_for_training':False,'updates':0,'steps_advanced':0,'teacher_batch_size':1})
    finally:env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['witness','run']);a=p.parse_args()
    assert not load(BASE/'process_offline.json')['completed']
    key='serial_witness' if a.stage=='witness' else 'offline_serial';record=BASE/f'process_{key}.json';assert not record.exists()
    assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
    snap=BASE/'serial_source'
    if not snap.exists():
        snap.mkdir()
        for name in ['bc_diagnosis.py','bc_diagnosis_serial.py','run_bc_diagnosis_serial.sh']:
            shutil.copyfile(ROOT/'scripts'/name,snap/name)
    save_json(BASE/f'provenance_{key}.json',{'source_sha256':sha(Path(__file__)),'config_sha256':sha(CONFIG),
              'checkpoint_sha256':sha(ROOT/PLAN['checkpoint']),'source_parent_sha256':sha(ROOT/'scripts/bc_diagnosis.py')})
    def timeout(sig,frame):raise TimeoutError('Remaining serial diagnostic budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(240 if a.stage=='witness' else 840);start=time.time()
    try:
        seed_all(0);witness() if a.stage=='witness' else run()
        assert sha(ROOT/PLAN['checkpoint'])==PLAN['checkpoint_sha256']
        save_json(record,{'completed':True,'wall_seconds':time.time()-start,'peak_torch_mib':torch.cuda.max_memory_allocated()/2**20})
    except BaseException as exc:
        save_json(record,{'completed':False,'wall_seconds':time.time()-start,'error':repr(exc)});raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
