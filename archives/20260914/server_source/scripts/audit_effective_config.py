"""Reconcile unused upstream trainer defaults with the actual custom pilot loop."""
import hashlib,json,os
from pathlib import Path
from omegaconf import OmegaConf
root=Path(os.environ['HAB_ROOT']);plan_path=root/'configs/state_pilot_v1.json';plan=json.loads(plan_path.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert "gym.make('PickSubtaskTrain-v0'" in (root/'scripts/hab_state.py').read_text()
used=['obs','obs_shape','action_dim','num_envs','num_eval_envs','episode_length','batch_size','model_size',
      'reward_coef','value_coef','consistency_coef','rho','lr','enc_lr_scale','grad_clip_norm','tau','discount',
      'mpc','iterations','num_samples','num_elites','num_pi_trajs','horizon','min_std','max_std','temperature',
      'log_std_min','log_std_max','entropy_coef','num_bins','vmin','vmax','bin_size','num_enc_layers','enc_dim',
      'mlp_dim','latent_dim','true_latent_dim','task_dim','num_q','dropout','simnorm_dim','multitask']
audits={}
for algo in ['tdmpc2','sac']:
    folder=root/f'artifacts/state_{algo}_pilot_v1';provenance=json.loads((folder/'provenance.json').read_text())
    summary=json.loads((folder/'summary.json').read_text())
    assert provenance['plan_sha256']==sha(plan_path)
    assert provenance['source_hashes']['scripts/hab_state.py']==sha(root/'scripts/hab_state.py')
    assert summary['train_steps']==plan['train_steps'] and summary['eval_episodes']==plan['eval_episodes']
    effective={'environment':{'env_id':'PickSubtaskTrain-v0','robot':'fetch','obs_mode':'state','control_mode':'pd_joint_delta_pos',
               'reward_mode':'normalized_dense','num_envs':1,'max_episode_steps':200,'active_action_dim':11,'state_dim':provenance['state_dim'],
               'input_additions':['current returned force /100','cumulative contact /5000','elapsed fraction'],
               'train_split':'train','eval_split':'val-development'},
               'training_plan':plan,'actual_train_steps':summary['train_steps'],'actual_updates':summary['updates'],
               'actual_eval_steps':summary['eval_steps'],'training_seed':plan['seed'],
               'basis':'Derived from unchanged, hash-audited custom entrypoint and completed episode/update logs; raw upstream initialization YAML is preserved.'}
    if algo=='tdmpc2':
        raw=OmegaConf.load(folder/'resolved_config.yaml');effective['algorithm']={k:OmegaConf.to_container(raw[k],resolve=True) if OmegaConf.is_config(raw[k]) else raw[k] for k in used}
        for k in ['batch_size','model_size','iterations','num_samples','num_elites','num_pi_trajs','horizon']:assert raw[k]==plan[k]
        effective['unused_upstream_trainer_defaults']={'env_id':raw.env_id,'steps':raw.steps,'seed':raw.seed,'eval_episodes_per_env':raw.eval_episodes_per_env,
               'explanation':'These fields are not used to construct HABState or schedule the custom training loop. They are not actual task/budget/seed values.'}
    else:
        effective['algorithm']={'hidden_dims':[256,256],'critic_count':2,'lr':.0003,'gamma':.975,'tau':.01,'batch_size':plan['batch_size'],'automatic_entropy_temperature':True}
    path=folder/'effective_config.json'
    assert not path.exists(),path
    path.write_text(json.dumps(effective,ensure_ascii=False,indent=2),encoding='utf-8')
    audits[algo]=effective
(root/'reports/state_pilot/effective_configs.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
report=root/'reports/state_pilot/报告.md'
with report.open('a',encoding='utf-8') as f:
    f.write('\n配置记录说明：TD-MPC2 原始 `resolved_config.yaml` 是上游算法初始化模板，其中未使用的环境名、训练步数、种子和评估数量仍为上游默认值。实际任务由 `HABState` 构造，预算与种子由 `state_pilot_v1.json` 决定；本轮真实运行 HAB Fetch Pick、10,000 步训练和 20 回合验证。原文件保留作审计，并新增 `effective_config.json` 与报告中的 `effective_configs.json` 明确本轮有效配置。该修正不改变训练轨迹、算法数值参数或结果。\n')
print('EFFECTIVE_CONFIG_AUDIT_PASS')
