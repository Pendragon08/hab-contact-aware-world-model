"""One seeded read-only diagnosis of visual/native state semantics, no training."""
import json
import torch
from bc_pipeline import ROOT, create_train_env, seed_all, native_state, schema
from legacy_evaluate_layouts import save_json, cpu_tree

out = ROOT/'artifacts/bc_v1/state_diagnosis'; out.mkdir(exist_ok=False)
seed_all(0); env = create_train_env()
try:
    seed_all(1199999)
    obs, info = env.reset(seed=1199999, options={'reconfigure': True, 'build_config_idxs': [6]})
    state, tree = native_state(env, info, 0, witness=True)
    teacher = obs['state'].float()
    result = {'teacher_shape': list(teacher.shape), 'native_shape': list(state.shape),
              'teacher': teacher.cpu().tolist(), 'native': state.cpu().tolist(),
              'fields': schema(env, tree),
              'differences': (teacher-state[:, :42]).cpu().tolist() if teacher.shape == (1,42) else 'shape differs',
              'obs_mode': env.unwrapped._obs_mode}
    save_json(out/'diagnosis.json', result)
    torch.save({'teacher_obs': cpu_tree(obs), 'native_tree': cpu_tree(tree), 'info': cpu_tree(info)}, out/'reads.pt')
    print('STATE_DIAGNOSIS', json.dumps(result), flush=True)
finally: env.close()
