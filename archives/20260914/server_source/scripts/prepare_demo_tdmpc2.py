"""Build auditable training-only transitions. CPU/NumPy; never launches training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ACTIVE = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
CHECKERS = ['is_grasped', 'ee_rest', 'robot_rest', 'is_static', 'cumulative_force_within_limit']


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def extract(root, row, source, train_spawns, dev_spawns, old_plan):
    folder = root / row['folder']
    idx = row['index'] if source == 'teacher' else row['collection_index']
    assert 0 <= idx < 80
    assert row['spawn_idx'] == old_plan['spawn_index_order'][idx]
    assert row['spawn_idx'] in train_spawns and row['spawn_idx'] not in dev_spawns
    assert row['seed'] == 1210000 + idx
    assert row['layout'] == 6 and row['task_idx'] == 456 and row['steps'] == 200
    expected = ('artifacts/bc_v1_2/collection' if source == 'teacher'
                else 'artifacts/bc_diagnosis_v1/fit') + f'/episode_{idx:03d}'
    assert row['folder'] == expected
    path = folder / 'trajectory.npz'
    assert sha(path) == row['trajectory_sha256'], str(path)
    events = [json.loads(x) for x in (folder / 'steps.jsonl').read_text().splitlines()]
    assert len(events) == 200
    assert [x['step'] for x in events] == list(range(1, 201))
    assert all(e['success'] == all(e[k] for k in CHECKERS) for e in events)
    assert any(e['success'] for e in events) == row['success_once']
    assert events[-1]['success'] == row['success_at_end']
    initial = load(folder / 'initial.json')['scene']
    assert initial['spawn_selection_idxs'] == [row['spawn_idx']]
    assert initial['task_plan']['subtasks'][0]['uid'] == 'tidy_house-pick-train-27436-0'
    assert initial['task_plan']['init_config_name'] == 'train/tidy_house/episode_5937.json'
    with np.load(path, allow_pickle=False) as z:
        state = z['states45'].copy()
        action = z['effective_actions11'].copy()
        assert state.shape == (201, 45) and action.shape == (200, 11)
        np.testing.assert_array_equal(action, z['executed_actions13'][:, ACTIVE])
        assert np.all(z['executed_actions13'][:, 8:10] == 0)
    rewards = np.asarray([e['reward'] for e in events], dtype=np.float32)
    assert abs(float(rewards.astype(np.float64).sum()) - row['return']) < 1e-5
    assert all(np.isfinite(x).all() for x in (state, action, rewards))
    assert np.abs(action).max() <= 1.00001
    np.testing.assert_allclose(state[:, 44], np.arange(201) / 200, atol=1e-7, rtol=0)
    np.testing.assert_allclose(state[1:, 42], [e['robot_force']/100 for e in events], atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(state[1:, 43], [e['robot_cumulative_force']/5000 for e in events], atol=1e-5, rtol=1e-6)
    labels = action.copy()
    actor_eligible = source == 'teacher' and row['success_at_end']
    provenance = {k: row[k] for k in ['folder', 'seed', 'spawn_idx', 'layout', 'task_idx']}
    provenance.update(source=source, collection_index=idx, trajectory_sha256=sha(path),
                      events_sha256=sha(folder/'steps.jsonl'), success_at_end=row['success_at_end'])
    if source == 'student':
        label_file = folder / 'shadow_teacher.npz'
        with np.load(label_file, allow_pickle=False) as z:
            raw = z['teacher_actions13']
            assert raw.shape == (200, 13) and np.isfinite(raw).all() and np.abs(raw).max() <= 1.00001
            labels = raw[:, ACTIVE].copy()
        provenance['correction_labels_sha256'] = sha(label_file)
    return state, action, rewards, labels, actor_eligible, provenance


def prepare(root, output):
    old_path = root/'configs/bc_v1_2.json'
    old = load(old_path)
    assert sha(old_path) == '7c2dc173dae9ecd06eda6a95d3cbe27079879c045a166993d163469cbb4483a6'
    teacher = load(root/'artifacts/bc_v1_2/collection/episodes.json')
    student = load(root/'artifacts/bc_diagnosis_v1/fit/episodes.json')
    assert len(teacher) == 100 and len(student) == 20
    train = teacher[:80]
    assert all(r['split'] == 'fit' and r['index'] == i for i, r in enumerate(train))
    assert all(r['split'] == 'dev' for r in teacher[80:])
    train_spawns = set(old['spawn_index_order'][:80]); dev_spawns = set(old['spawn_index_order'][80:])
    assert len(train_spawns) == 80 and len(dev_spawns) == 20 and train_spawns.isdisjoint(dev_spawns)
    assert len({r['collection_index'] for r in student}) == 20
    rows = [extract(root, r, source, train_spawns, dev_spawns, old)
            for source, group in [('teacher', train), ('student', student)] for r in group]
    states, actions, rewards, labels = [np.stack([r[i] for r in rows]) for i in range(4)]
    actor_mask = np.asarray([r[4] for r in rows], dtype=bool)
    assert actor_mask.sum() == 53
    done = np.zeros((100, 200), dtype=bool); done[:, -1] = True
    # Explicit finite horizon: success is an event, not an early terminal.
    returns = np.zeros_like(rewards, dtype=np.float64)
    for t in range(199, -1, -1):
        returns[:, t] = rewards[:, t] + (0.975 * returns[:, t+1] if t < 199 else 0)
    # Fit statistics contain pre-action training states only, no development observations.
    fitting = states[:, :-1].reshape(-1, 45).astype(np.float64)
    mean = fitting.mean(axis=0).astype(np.float32)
    std = np.maximum(fitting.std(axis=0), 1e-3).astype(np.float32)
    # Keep time explicit in [0,1] for the finite-horizon adapter and planning mask.
    mean[44] = 0; std[44] = 1
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'training.npz', observations=states, actions=actions, rewards=rewards,
                        terminal=done, monte_carlo_returns=returns.astype(np.float32),
                        actor_labels=labels, successful_teacher_episode=actor_mask,
                        correction_episode=np.arange(100) >= 80, mean=mean, std=std)
    report = dict(completed=True, training_started=False, episodes=100, transitions=20000,
                  teacher_transitions=16000, student_transitions=4000,
                  successful_teacher_actor_transitions=10600, correction_actor_transitions=4000,
                  development_transitions_in_training=0, development_labels_in_training=0,
                  finite_horizon=200, discount=0.975, terminal_masks=100,
                  terminal_policy='Only t=200 terminal; keep all early-success transitions.',
                  action_semantics='Actions are executed controls. Shadow labels are actor supervision only.',
                  train_spawns=sorted(train_spawns), development_spawns=sorted(dev_spawns),
                  normalizer='All 20000 pre-action training states; population std floor 1e-3; time unchanged.',
                  payload_sha256=sha(output/'training.npz'), payload_bytes=(output/'training.npz').stat().st_size,
                  source_code_sha256=sha(Path(__file__)), parent_config_sha256=sha(old_path),
                  sources=[r[5] for r in rows])
    write(output/'manifest.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    prepare(a.root, a.output or a.root/'artifacts/demo_tdmpc2_v1/data')
