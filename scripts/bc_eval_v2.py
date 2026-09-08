"""Continue 19 fixed-checkpoint evaluations after validating semantic initial-state equivalence."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from bc_pipeline import ROOT,BASE,PLAN,PLAN_PATH,BC,ACTIVE,create_train_env,rollout,teacher_actor,load_policy,selection_path,seed_all,sha
from legacy_evaluate_layouts import save_json


def canonical_pair(teacher,student):
    a=json.loads((teacher/'initial.json').read_text()); b=json.loads((student/'initial.json').read_text())
    # Upstream converts a set of init-config names into process-local integer indices.
    sa={k:v for k,v in a['scene'].items() if k!='init_config_idxs'}
    sb={k:v for k,v in b['scene'].items() if k!='init_config_idxs'}
    assert sa==sb, 'Semantic scene/task/spawn mismatch'
    x=torch.load(teacher/'initial_snapshot.pt',map_location='cpu',weights_only=False)
    y=torch.load(student/'initial_snapshot.pt',map_location='cpu',weights_only=False)
    parked=[];differences=[]
    def visit(av,bv,path):
        if isinstance(av,dict):
            assert av.keys()==bv.keys(),path
            for k in av:
                if path=='environment' and k=='init_config_idxs':continue
                visit(av[k],bv[k],path+'.'+str(k))
        elif isinstance(av,(list,tuple)):
            assert len(av)==len(bv),path
            for i,(aa,bb) in enumerate(zip(av,bv)):visit(aa,bb,path+f'[{i}]')
        elif isinstance(av,(torch.Tensor,np.ndarray)):
            aa=av.numpy() if isinstance(av,torch.Tensor) else av
            bb=bv.numpy() if isinstance(bv,torch.Tensor) else bv
            assert aa.shape==bb.shape,path
            if path.startswith('environment.actors.') and aa.shape==(1,13):
                pa=bool(np.all(aa[0,:2]==-10000) and aa[0,2]<-9990)
                pb=bool(np.all(bb[0,:2]==-10000) and bb[0,2]<-9990)
                assert pa==pb, 'Active/hidden actor mismatch: '+path
                if pa:
                    parked.append({'actor':path,'teacher_z':float(aa[0,2]),'student_z':float(bb[0,2])})
                    aa=aa.copy();bb=bb.copy();aa[:,2]=bb[:,2]=-10000
            diff=float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
            assert diff<=1e-5,(path,diff)
            differences.append({'path':path,'max_abs_difference':diff})
        else:assert av==bv,(path,av,bv)
    for k in ['environment','controller','observation']:visit(x[k],y[k],k)
    assert a['observation_sha256']==b['observation_sha256'], 'Initial teacher input must remain bit-identical'
    return {'passed':True,'initial_observation_sha256':a['observation_sha256'],
            'semantic_init_name':sa['task_plan']['init_config_name'],
            'process_local_init_indices':[a['scene']['init_config_idxs'],b['scene']['init_config_idxs']],
            'parked_actors':parked,'checks':differences,
            'note':'Only process-local init indices and the z parking slot of actors whose x/y=-10000 are canonicalized. All active actors, robot/controller and full observations compared.'}


def witness():
    folder=BASE/'eval_witness_v2'; folder.mkdir(exist_ok=False)
    original=BASE/'student/evaluation/episode_080';teacher=BASE/'collection/episode_080'
    save_json(folder/'retained_student_initial_audit.json',canonical_pair(teacher,original))
    env=create_train_env()
    try:
        policy,digest=load_policy(env);assert digest==PLAN['checkpoint_sha256']
        sel=json.loads(selection_path().read_text())
        row=rollout(env,teacher_actor(policy),folder/'teacher_prefix',1210080,sel['layout'],sel['task_idx'],steps=8,spawn_idx=26)
        save_json(folder/'teacher_initial_audit.json',canonical_pair(teacher,ROOT/row['folder']))
        a=np.load(teacher/'trajectory.npz');b=np.load(ROOT/row['folder']/'trajectory.npz')
        diffs={k:float(np.max(np.abs(a[k][:len(b[k])].astype(float)-b[k].astype(float)))) for k in b.files}
        assert all(v<=1e-5 for v in diffs.values()),diffs
        ea=[json.loads(s) for s in (teacher/'steps.jsonl').read_text().splitlines()][:8]
        eb=[json.loads(s) for s in (ROOT/row['folder']/'steps.jsonl').read_text().splitlines()]
        assert ea==eb
        save_json(folder/'complete.json',{'passed':True,'plan_sha256':sha(PLAN_PATH),'prefix_differences':diffs,'retained_first_student_failure':True})
        print('BC_EVAL_CANONICAL_WITNESS_PASS',diffs,flush=True)
    finally:env.close()


def run():
    assert json.loads((BASE/'eval_witness_v2/complete.json').read_text())['passed']
    original_process=json.loads((BASE/'process_train.json').read_text())
    assert not original_process['completed'] and 'start mismatch' in original_process['error']
    folder=BASE/'student/evaluation_v2';folder.mkdir(exist_ok=False)
    teacher_rows=json.loads((BASE/'collection/episodes.json').read_text())[80:]
    rows=[json.loads((BASE/'student/evaluation/episode_080/episode.json').read_text())]
    assert rows[0]['seed']==1210080 and rows[0]['steps']==200
    save_json(folder/'initial_audit_080.json',canonical_pair(ROOT/teacher_rows[0]['folder'],ROOT/rows[0]['folder']))
    checkpoint_path=BASE/'student/checkpoint_05000.pt'; checkpoint_sha=sha(checkpoint_path)
    ckpt=torch.load(checkpoint_path,map_location='cuda',weights_only=False);assert ckpt['update']==5000
    model=BC(ckpt['model']['mean'],ckpt['model']['std']).cuda();model.load_state_dict(ckpt['model']);model.eval()
    def actor(obs,state):
        full=torch.zeros(1,13,device='cuda');full[:,ACTIVE]=model(state);return full
    env=create_train_env();sel=json.loads(selection_path().read_text())
    try:
        for item in teacher_rows[1:]:
            result=rollout(env,actor,folder/f"episode_{item['index']:03d}",item['seed'],sel['layout'],sel['task_idx'],spawn_idx=item['spawn_idx'])
            save_json(folder/f"initial_audit_{item['index']:03d}.json",canonical_pair(ROOT/item['folder'],ROOT/result['folder']))
            rows.append(result);save_json(BASE/'student/episodes.json',rows)
        assert sha(checkpoint_path)==checkpoint_sha and len(rows)==20
        counts={k:sum(r[k] for r in rows) for k in ['ever_grasp','success_once','success_at_end']}
        passed=counts['ever_grasp']>=10 and counts['success_once']>=5
        final_metrics=json.loads((BASE/'student/metrics.json').read_text())[-1]
        save_json(BASE/'student/complete.json',{'passed':passed,'episodes':20,'counts':counts,'training_updates':5000,
                  'plan_sha256':sha(PLAN_PATH),'checkpoint_sha256':checkpoint_sha,'final_supervised_metrics':final_metrics,
                  'recovery':'First completed failure retained; only remaining 19 starts evaluated with unchanged final checkpoint. Canonical initial checks replace process-local index/parking hashes.'})
        print('BC_STUDENT_COMPLETE',passed,counts,flush=True)
    finally:env.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['witness','run']);a=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    dest=BASE/f'eval_v2_{a.mode}_process.json';assert not dest.exists()
    save_json(BASE/f'eval_v2_{a.mode}_provenance.json',{'source_sha256':sha(Path(__file__)),
              'checkpoint_sha256':sha(BASE/'student/checkpoint_05000.pt'),'plan_sha256':sha(PLAN_PATH)})
    def timeout(sig,frame):raise TimeoutError('Continuation time budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300 if a.mode=='witness' else 3200)
    start=time.time()
    try:
        seed_all(0)
        witness() if a.mode=='witness' else run()
        result={'completed':True,'wall_seconds':time.time()-start,'peak_torch_mib':torch.cuda.max_memory_allocated()/2**20}
        save_json(dest,result)
        if a.mode=='run':
            original=json.loads((BASE/'process_train.json').read_text())
            save_json(BASE/'process_train_completed.json',{'completed':True,'wall_seconds':original['wall_seconds']+result['wall_seconds'],
                      'peak_torch_mib':None,'continuation_peak_torch_mib':result['peak_torch_mib'],
                      'note':'Original training peak unavailable after assertion; measured continuation separately. Original failed process metadata retained.'})
    except BaseException as exc:
        save_json(dest,{'completed':False,'wall_seconds':time.time()-start,'error':repr(exc)});raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
