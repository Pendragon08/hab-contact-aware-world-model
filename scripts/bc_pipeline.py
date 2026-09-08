"""Versioned teacher/state pairing, frozen demonstrations and closed-loop BC pilot."""
import argparse
import dataclasses
import json
import os
import random
import shutil
import signal
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from mani_skill import ASSET_DIR
from mani_skill.utils import common
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from demo import load_policy, frame
from hab_state import ACTIVE
from contact_experiment import digest_tree
from legacy_evaluate_layouts import CHECKERS, FLOATS, cpu_tree, save_json, sha

ROOT = Path(os.environ['HAB_ROOT'])
PLAN_PATH = ROOT / os.environ.get('HAB_BC_CONFIG', 'configs/bc_v1.json')
PLAN = json.loads(PLAN_PATH.read_text())
BASE = ROOT / 'artifacts' / PLAN['version']
MOTION = [i for i in range(11) if i != 7]


def interface_path(): return ROOT / PLAN.get('interface_source', str((BASE/'interface').relative_to(ROOT)))
def selection_path(): return ROOT / PLAN.get('selection_source', str((BASE/'selection.json').relative_to(ROOT)))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def create_train_env():
    root = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange'
    cfg = EnvConfig(env_id='PickSubtaskTrain-v0', num_envs=1, max_episode_steps=200,
                    obs_mode='depth', render_mode='rgb_array', sim_backend='gpu',
                    record_video=False, frame_stack=3, continuous_task=True,
                    task_plan_fp=str(root/'task_plans/tidy_house/pick/train/all.json'),
                    spawn_data_fp=str(root/'spawn_data/tidy_house/pick/train/spawn_data.pt'),
                    env_kwargs={'require_build_configs_repeated_equally_across_envs': False,
                                'add_event_tracker_info': True})
    env = make_env(cfg); env.env.auto_reset = False
    mapping = {k: list(v) for k, v in env.unwrapped.agent.controller.action_mapping.items()}
    assert mapping == {'arm': [0, 7], 'gripper': [7, 8], 'body': [8, 11], 'base': [11, 13]}
    return env


def native_state(env, info, step, witness=False):
    """Read the upstream native state dictionary without evaluating or advancing physics."""
    u = env.unwrapped
    tree = u._get_obs_state_dict(info)
    raw = common.flatten_state_dict(tree, use_torch=True, device=u.device).float()
    assert raw.shape == (1, 42)
    if witness:
        before = digest_tree({'env': u.get_state_dict(), 'controller': u.agent.controller.get_state(),
                              'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
        original_mode = u._obs_mode
        try:
            u._obs_mode = 'state'
            reference = u.get_obs(info)
        finally:
            u._obs_mode = original_mode
        assert torch.equal(raw, reference.float()), 'Native state path mismatch'
        after = digest_tree({'env': u.get_state_dict(), 'controller': u.agent.controller.get_state(),
                             'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
        assert before == after, 'Native state read changed simulation/RNG'
    result = torch.cat([raw, info['robot_force'].reshape(1, 1)/100.,
                        info['robot_cumulative_force'].reshape(1, 1)/5000.,
                        torch.full((1, 1), step/200., device=u.device)], dim=1)
    assert result.shape == (1, 45) and torch.isfinite(result).all()
    return result, tree


def schema(env, tree):
    fields = []; offset = 0
    def visit(value, prefix):
        nonlocal offset
        if isinstance(value, dict):
            for key, child in value.items(): visit(child, prefix+[key])
        else:
            n = int(value[0].numel())
            fields.append({'field': '.'.join(prefix), 'start': offset, 'stop_exclusive': offset+n})
            offset += n
    visit(tree, [])
    assert offset == 42
    fields += [{'field': 'robot_force/100', 'start': 42, 'stop_exclusive': 43},
               {'field': 'robot_cumulative_force/5000', 'start': 43, 'stop_exclusive': 44},
               {'field': 'elapsed_steps/200', 'start': 44, 'stop_exclusive': 45}]
    return {'fields': fields, 'active_joint_order': [j.name for j in env.unwrapped.agent.robot.active_joints][3:],
            'units': {'qpos': 'revolute rad, prismatic m (joint order recorded)',
                      'qvel': 'revolute rad/s, prismatic m/s',
                      'pose': 'xyz in m then quaternion wxyz, relative to base',
                      'goal_pos': 'm, relative to base', 'is_grasped': 'boolean 0/1',
                      'force': 'upstream contact-force aggregate, scaled /100; cumulative aggregate /5000, not an impulse'},
            'action_indices_in_raw13': ACTIVE, 'gripper_student_index': 7,
            'gripper_semantics': 'normalized absolute opening; not a delta',
            'head_raw13_indices': [8, 9], 'head_execution': 'zero',
            'depth_storage': 'initial full 3-frame stacks in snapshot; newest frame at all 201 observations',
            'state_scope': 'privileged object pose, no full obstacle geometry'}


def scalar(info, key): return float(info[key][0].item())


def rollout(env, actor, target, seed, layout, task_idx=None, dual=True, steps=200, video=True, spawn_idx=None):
    if shutil.disk_usage(ROOT).free < 15*2**30: raise RuntimeError('Remote reserve below 15 GiB')
    target.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    options = {'reconfigure': True, 'build_config_idxs': [layout]}
    if task_idx is not None: options['task_plan_idxs'] = torch.tensor([task_idx], dtype=torch.int)
    if spawn_idx is not None: options['spawn_selection_idxs'] = [spawn_idx]
    obs, info = env.reset(seed=seed, options=options)
    u = env.unwrapped
    assert list(u.build_config_idxs) == [layout]
    selected = int(u.task_plan_idxs[0])
    if task_idx is not None: assert selected == task_idx
    if spawn_idx is not None: assert list(u.spawn_selection_idxs) == [spawn_idx]
    task = u.build_config_idx_to_task_plans[layout][selected]
    scene = {'layout': layout, 'task_idx': selected, 'init_config_idxs': list(u.init_config_idxs),
             'spawn_selection_idxs': list(u.spawn_selection_idxs),
             'target_object': u.subtask_objs[0]._objs[0].name,
             'task_plan': dataclasses.asdict(task)}
    snapshot = {'environment': cpu_tree(u.get_state_dict()), 'controller': cpu_tree(u.agent.controller.get_state()),
                'observation': cpu_tree(obs), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
    torch.save(snapshot, target/'initial_snapshot.pt')
    save_json(target/'initial.json', {'scene': scene, 'environment_sha256': digest_tree(snapshot['environment']),
                                   'controller_sha256': digest_tree(snapshot['controller']),
                                   'observation_sha256': digest_tree(snapshot['observation'])})
    data = {'teacher_states': [], 'raw_actions13': [], 'executed_actions13': [], 'effective_actions11': []}
    if dual: data['states45'] = []
    events = []; physical = []; started = time.time()
    def record(step, obs, info):
        data['teacher_states'].append(obs['state'][0].detach().cpu().numpy().copy())
        for key, value in obs['pixels'].items():
            data.setdefault(key, []).append(value[0, -1].detach().cpu().numpy().copy())
        physical.append(torch.cat([u.agent.robot.qpos, u.agent.robot.qvel,
                                   u.subtask_objs[0].pose.raw_pose, u.agent.tcp_pose.raw_pose], dim=1)[0].cpu().numpy().copy())
        if dual:
            x, tree = native_state(env, info, step)
            assert float((x[:, :42]-obs['state'].float()).abs().max()) <= 1e-5, 'Teacher/state feature alignment differs'
            data['states45'].append(x[0].cpu().numpy().copy())
            if step == 0: save_json(target/'schema.json', schema(env, tree))
            return x
        return None
    writer = imageio.get_writer(target/'rollout.mp4', fps=20, codec='libx264', quality=6) if video else None
    state = record(0, obs, info)
    if writer: writer.append_data(frame(env))
    try:
        with (target/'steps.jsonl').open('x') as stream:
            for step in range(1, steps+1):
                with torch.no_grad(): raw = actor(obs, state)
                assert raw.shape == (1, 13) and torch.isfinite(raw).all() and raw.abs().max() <= 1.00001
                full = raw.clone(); full[:, 8:10] = 0
                data['raw_actions13'].append(raw[0].cpu().numpy().copy())
                data['executed_actions13'].append(full[0].cpu().numpy().copy())
                data['effective_actions11'].append(full[0, ACTIVE].cpu().numpy().copy())
                obs, reward, term, trunc, info = env.step(full.clone())
                current = info.get('final_info', info)
                success = bool(info['episode']['s_e'][0]) if bool((term|trunc).any()) else bool(current['success'][0])
                event = {'step': step, 'reward': float(reward[0]), 'success': success}
                event.update({k: bool(current[k][0]) for k in CHECKERS})
                event.update({k: scalar(current, k) for k in FLOATS})
                assert success == all(event[k] for k in CHECKERS)
                assert bool(trunc[0]) == (step == 200)
                stream.write(json.dumps(event)+'\n'); stream.flush(); events.append(event)
                state = record(step, obs, current)
                if writer: writer.append_data(frame(env))
                if step == 200:
                    terminal = {k: v[0].item() for k, v in info['episode'].items()}
                    assert terminal['l'] == 200 and bool(terminal['s_o']) == any(e['success'] for e in events)
                    save_json(target/'terminal.json', terminal)
    finally:
        if writer: writer.close()
        arrays = {k: np.stack(v) for k, v in data.items() if v}
        arrays['physical_states'] = np.stack(physical)
        np.savez_compressed(target/'trajectory.npz', **arrays)
    assert len(events) == steps
    row = {'seed': seed, 'layout': layout, 'task_idx': selected, 'target_object': scene['target_object'],
           'spawn_idx': int(u.spawn_selection_idxs[0]),
           'steps': len(events), 'success_once': any(e['success'] for e in events),
           'success_at_end': events[-1]['success'], 'ever_grasp': any(e['is_grasped'] for e in events),
           'return': sum(e['reward'] for e in events), 'wall_seconds': time.time()-started,
           'final_conditions': {k: events[-1][k] for k in CHECKERS},
           'trajectory_sha256': sha(target/'trajectory.npz'), 'folder': target.relative_to(ROOT).as_posix()}
    save_json(target/'episode.json', row)
    print('EPISODE_COMPLETE', json.dumps(row), flush=True)
    return row


def teacher_actor(policy):
    def act(obs, state):
        obs = to_tensor(obs, device='cuda', dtype='float')
        return policy.actor(obs['pixels'], obs['state'], compute_pi=False, compute_log_pi=False)[0]
    return act


def compare(a, b):
    pa = ROOT/a['folder']; pb = ROOT/b['folder']
    assert json.loads((pa/'initial.json').read_text()) == json.loads((pb/'initial.json').read_text())
    x = np.load(pa/'trajectory.npz'); y = np.load(pb/'trajectory.npz')
    diffs = {k: float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))) for k in x.files}
    ea = [json.loads(s) for s in (pa/'steps.jsonl').read_text().splitlines()]
    eb = [json.loads(s) for s in (pb/'steps.jsonl').read_text().splitlines()]
    boolean_equal = all(all(aa[k] == bb[k] for k in CHECKERS+['success']) for aa, bb in zip(ea, eb))
    diffs['event_numeric'] = max(abs(aa[k]-bb[k]) for aa, bb in zip(ea, eb) for k in FLOATS+['reward'])
    result = {'seed': a['seed'], 'passed': boolean_equal and all(v <= 1e-5 for v in diffs.values()),
              'max_abs_difference': diffs, 'boolean_equal': boolean_equal}
    save_json(pb/'paired_audit.json', result)
    assert result['passed'], result
    return result


class BC(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean); self.register_buffer('std', std)
        self.net = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 11), nn.Tanh())
    def forward(self, state): return self.net((state-self.mean)/self.std)


def losses(pred, target):
    motion = (pred[:, MOTION]-target[:, MOTION]).square().mean()
    gripper = (pred[:, 7]-target[:, 7]).square().mean()
    return motion+gripper, motion, gripper


def interface(env, actor, witness):
    out = BASE/('witness' if witness else 'interface'); out.mkdir(parents=True, exist_ok=False)
    rows = []; audits = []
    seeds = [PLAN['witness_seed']] if witness else PLAN['interface_seeds']
    for i, seed in enumerate(seeds):
        layout = PLAN['layouts'][i//4]
        spawn = 0 if witness and 'spawn_index_order' in PLAN else None
        a = rollout(env, actor, out/f'pair_{i:02d}/baseline', seed, layout, dual=False, steps=8 if witness else 200, spawn_idx=spawn)
        b = rollout(env, actor, out/f'pair_{i:02d}/dual', seed, layout, dual=True, steps=8 if witness else 200, spawn_idx=spawn)
        rows.append(a); audits.append(compare(a, b))
        save_json(out/'progress.json', {'rows': rows, 'audits': audits})
    if witness:
        obs, info = env.reset(seed=PLAN['witness_seed'], options={'reconfigure': True, 'build_config_idxs': [6]})
        x, tree = native_state(env, info, 0, witness=True)
        model = BC(torch.zeros(45, device='cuda'), torch.ones(45, device='cuda')).cuda()
        optim = torch.optim.Adam(model.parameters(), lr=PLAN['learning_rate'])
        loss, _, _ = losses(model(x), torch.zeros(1, 11, device='cuda'))
        loss.backward(); optim.step()
        assert torch.isfinite(loss)
        save_json(out/'schema.json', schema(env, tree))
        save_json(out/'complete.json', {'passed': True, 'plan_sha256': sha(PLAN_PATH), 'loss': float(loss), 'pairs': audits})
        print('BC_DOC_WITNESS_PASS', float(loss), torch.cuda.get_device_name(), flush=True)
    else:
        eligible = [r for r in rows if r['success_at_end']]
        result = {'passed': True, 'pairs': audits, 'rows': rows, 'plan_sha256': sha(PLAN_PATH)}
        save_json(out/'complete.json', result)
        if not eligible: raise RuntimeError('No teacher-solvable training task; stop before collection')
        selected = eligible[0]
        save_json(BASE/'selection.json', {'rule': PLAN['selection_rule'], 'source_episode': selected,
                                         'scene': json.loads((ROOT/selected['folder']/'initial.json').read_text())['scene'],
                                         'layout': selected['layout'], 'task_idx': selected['task_idx']})


def collect(env, actor):
    assert json.loads((interface_path()/'complete.json').read_text())['passed']
    selection = json.loads(selection_path().read_text())
    out = BASE/'collection'; out.mkdir(exist_ok=False)
    order = PLAN['spawn_index_order']; assert sorted(order) == list(range(100))
    manifest = [{'index': i, 'seed': PLAN['collection_seed_base']+i, 'split': 'fit' if i < PLAN['fit_episodes'] else 'dev', 'spawn_idx': order[i]}
                for i in range(PLAN['collection_episodes'])]
    save_json(out/'split_manifest.json', manifest)
    rows = []
    for item in manifest:
        r = rollout(env, actor, out/f"episode_{item['index']:03d}", item['seed'], selection['layout'], selection['task_idx'], spawn_idx=item['spawn_idx'])
        initial = json.loads((ROOT/r['folder']/'initial.json').read_text())['scene']
        assert initial['task_plan'] == selection['scene']['task_plan']
        r.update(item); rows.append(r); save_json(out/'episodes.json', rows)
    fit = [r for r in rows if r['split'] == 'fit' and r['success_at_end']]
    save_json(out/'complete.json', {'passed': len(fit) >= 10, 'episodes': len(rows), 'fit_success_at_end': len(fit),
                                   'selected_fit_seeds': [r['seed'] for r in fit], 'plan_sha256': sha(PLAN_PATH)})
    if len(fit) < 10: raise RuntimeError('Fewer than 10 successful fit demos; BC not launched')


def train(env):
    assert json.loads((BASE/'collection/complete.json').read_text())['passed']
    rows = json.loads((BASE/'collection/episodes.json').read_text())
    fit = [r for r in rows if r['split'] == 'fit' and r['success_at_end']]
    dev = [r for r in rows if r['split'] == 'dev']
    assert len(fit) >= 10 and len(dev) == 20 and not ({r['seed'] for r in fit}&{r['seed'] for r in dev})
    assert not ({r['spawn_idx'] for r in fit}&{r['spawn_idx'] for r in dev})
    def dataset(items):
        xs = []; ys = []
        for r in items:
            p = ROOT/r['folder']/'trajectory.npz'; assert sha(p) == r['trajectory_sha256']
            with np.load(p) as d:
                assert d['states45'].shape == (201, 45) and d['effective_actions11'].shape == (200, 11)
                xs.append(d['states45'][:-1]); ys.append(d['effective_actions11'])
        return torch.tensor(np.concatenate(xs), device='cuda'), torch.tensor(np.concatenate(ys), device='cuda')
    x, y = dataset(fit); xd, yd = dataset(dev)
    initial_fit = x[::200, :42]; initial_dev = xd[::200, :42]
    assert ((initial_fit[:, None]-initial_dev[None]).abs().amax(-1) > 1e-5).all(), 'Duplicate initial state across fit/dev'
    seed_all(PLAN['training_seed'])
    out = BASE/'student'; out.mkdir(exist_ok=False)
    mean = x.mean(0); std = x.std(0, unbiased=False).clamp_min(1e-3)
    model = BC(mean, std).cuda(); optimizer = torch.optim.Adam(model.parameters(), lr=PLAN['learning_rate'])
    save_json(out/'training_manifest.json', {'fit': fit, 'dev': dev, 'normalization_sources': [r['seed'] for r in fit],
                                          'normalizer_mean': mean.cpu().tolist(), 'normalizer_std': std.cpu().tolist(),
                                          'parameters': sum(p.numel() for p in model.parameters()), 'plan_sha256': sha(PLAN_PATH)})
    def evaluate(xv, yv):
        with torch.no_grad(): values = losses(model(xv), yv)
        return dict(zip(['total', 'motion', 'gripper'], [float(v) for v in values]))
    metrics = [{'update': 0, 'fit': evaluate(x, y), 'dev': evaluate(xd, yd), 'first_fit_trajectory': evaluate(x[:200], y[:200])}]
    with (out/'updates.jsonl').open('x') as stream:
        for update in range(1, PLAN['updates']+1):
            idx = torch.randint(len(x), (PLAN['batch_size'],), device='cuda')
            loss, _, _ = losses(model(x[idx]), y[idx]); assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            stream.write(json.dumps({'update': update, 'loss': float(loss)})+'\n')
            if update % 250 == 0:
                metric = {'update': update, 'fit': evaluate(x, y), 'dev': evaluate(xd, yd),
                          'first_fit_trajectory': evaluate(x[:200], y[:200])}
                metrics.append(metric); save_json(out/'metrics.json', metrics)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'update': update,
                            'plan': PLAN, 'training_manifest_sha256': sha(out/'training_manifest.json')}, out/f'checkpoint_{update:05d}.pt')
                print('BC_UPDATE', json.dumps(metric), flush=True)
    model.eval()
    def act(obs, state):
        full = torch.zeros(1, 13, device='cuda'); full[:, ACTIVE] = model(state); return full
    selection = json.loads(selection_path().read_text()); results = []
    for r in dev:
        result = rollout(env, act, out/f"evaluation/episode_{r['index']:03d}", r['seed'], selection['layout'], selection['task_idx'], spawn_idx=r['spawn_idx'])
        student_initial = json.loads((ROOT/result['folder']/'initial.json').read_text())
        teacher_initial = json.loads((ROOT/r['folder']/'initial.json').read_text())
        assert student_initial == teacher_initial, 'Teacher/student start mismatch'
        results.append(result); save_json(out/'episodes.json', results)
    counts = {key: sum(r[key] for r in results) for key in ['ever_grasp', 'success_once', 'success_at_end']}
    passed = counts['ever_grasp'] >= PLAN['gate']['ever_grasp_count'] and counts['success_once'] >= PLAN['gate']['success_once_count']
    save_json(out/'complete.json', {'passed': passed, 'episodes': len(results), 'counts': counts, 'training_updates': PLAN['updates'],
                                   'plan_sha256': sha(PLAN_PATH), 'final_supervised_metrics': metrics[-1],
                                   'meaning': 'single training task development gate; trained BC, not world model or risk planning'})
    print('BC_STUDENT_COMPLETE', passed, counts, flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['witness', 'interface', 'collect', 'train']); args = p.parse_args()
    assert not (args.stage == 'interface' and 'interface_source' in PLAN), 'Use recorded R001; do not rerun it under learning protocol'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    BASE.mkdir(parents=True, exist_ok=True)
    def timeout(signum, frame): raise TimeoutError('Stage time budget reached; preserve partial attempt')
    signal.signal(signal.SIGALRM, timeout); signal.signal(signal.SIGTERM, timeout)
    signal.alarm(PLAN['stage_seconds'][args.stage])
    started = time.time(); env = None
    provenance = BASE/f'provenance_{args.stage}.json'
    assert not provenance.exists(), 'Refusing to overwrite an attempted stage'
    save_json(provenance, {'stage': args.stage, 'plan_sha256': sha(PLAN_PATH), 'started_unix': started,
                          'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'torch': torch.__version__,
                          'source_hashes': {p.name: sha(p) for p in (ROOT/'scripts').glob('*.py')}})
    try:
        if args.stage != 'witness':
            witness = json.loads((BASE/'witness/complete.json').read_text())
            assert witness['passed'] and witness['plan_sha256'] == sha(PLAN_PATH)
        seed_all(0); env = create_train_env()
        if args.stage == 'train': train(env)
        else:
            policy, digest = load_policy(env); assert digest == PLAN['checkpoint_sha256']
            actor = teacher_actor(policy)
            if args.stage == 'collect': collect(env, actor)
            else: interface(env, actor, args.stage == 'witness')
        save_json(BASE/f'process_{args.stage}.json', {'completed': True, 'wall_seconds': time.time()-started,
                                                   'peak_torch_mib': torch.cuda.max_memory_allocated()/2**20})
    except BaseException as exc:
        save_json(BASE/f'process_{args.stage}.json', {'completed': False, 'wall_seconds': time.time()-started,
                                                   'error': repr(exc)})
        raise
    finally:
        signal.alarm(0)
        if env is not None: env.close()


if __name__ == '__main__': main()
