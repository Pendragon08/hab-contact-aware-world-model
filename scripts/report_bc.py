"""Recompute BC counts, alignment, normalization, action replay and videos from raw files."""
import json
import os
from collections import Counter
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bc_pipeline import BASE, PLAN, BC, ACTIVE, CHECKERS, sha, interface_path, selection_path
from bc_eval_v2 import canonical_pair

ROOT = Path(os.environ['HAB_ROOT'])
OUT = ROOT/'reports'/PLAN['version']
def save(path, data): path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
def load(path): return json.loads(path.read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    interface = load(interface_path()/'complete.json')
    assert interface['passed'] and len(interface['pairs']) == 20
    collection = load(BASE/'collection/episodes.json')
    student = load(BASE/'student/episodes.json')
    complete = load(BASE/'student/complete.json')
    assert len(collection) == 100 and len(student) == 20
    fit = [r for r in collection if r['split'] == 'fit' and r['success_at_end']]
    dev = [r for r in collection if r['split'] == 'dev']
    assert len(fit) >= 10 and [r['seed'] for r in dev] == list(range(1210080, 1210100))
    assert not ({r['spawn_idx'] for r in fit}&{r['spawn_idx'] for r in dev})
    checkpoint = torch.load(BASE/'student/checkpoint_05000.pt', map_location='cpu', weights_only=False)
    model = BC(torch.zeros(45), torch.ones(45)); model.load_state_dict(checkpoint['model']); model.eval()
    norm = load(BASE/'student/training_manifest.json')
    assert norm['normalization_sources'] == [r['seed'] for r in fit]
    states = np.concatenate([np.load(ROOT/r['folder']/'trajectory.npz')['states45'][:-1] for r in fit])
    # Compare with torch's train-only reduction, allowing CPU/GPU summation order.
    x = torch.from_numpy(states)
    torch.testing.assert_close(x.mean(0), model.mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.std(0, unbiased=False).clamp_min(1e-3), model.std, rtol=1e-5, atol=1e-5)
    updates = [json.loads(s) for s in (BASE/'student/updates.jsonl').read_text().splitlines()]
    assert len(updates) == 5000 and all(r['update'] == i+1 and np.isfinite(r['loss']) for i, r in enumerate(updates))
    video_counts = []; replay_error = 0.; endings = Counter()
    all_rows = interface['rows'] + collection + student
    # Baseline and dual interface runs are separately audited; include all of them here.
    all_rows += [load(interface_path()/f'pair_{i:02d}/dual/episode.json') for i in range(20)]
    for row in all_rows:
        folder = ROOT/row['folder']; d = np.load(folder/'trajectory.npz')
        events = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(events) == row['steps'] == 200
        assert sha(folder/'trajectory.npz') == row['trajectory_sha256']
        assert all(np.isfinite(d[k]).all() for k in d.files)
        assert d['teacher_states'].shape == (201, 42)
        np.testing.assert_array_equal(d['effective_actions11'], d['executed_actions13'][:, ACTIVE])
        assert np.max(np.abs(d['executed_actions13'][:, 8:10])) == 0
        assert row['success_once'] == any(e['success'] for e in events)
        assert row['success_at_end'] == events[-1]['success']
        assert row['ever_grasp'] == any(e['is_grasped'] for e in events)
        assert all(e['success'] == all(e[k] for k in CHECKERS) for e in events)
        if 'states45' in d:
            np.testing.assert_allclose(d['states45'][:, :42], d['teacher_states'], rtol=0, atol=1e-5)
            np.testing.assert_allclose(d['states45'][:, 44], np.arange(201)/200, atol=1e-7)
            np.testing.assert_allclose(d['states45'][1:, 42], [e['robot_force']/100 for e in events], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(d['states45'][1:, 43], [e['robot_cumulative_force']/5000 for e in events], rtol=1e-6, atol=1e-6)
        if '/student/' in row['folder']:
            with torch.no_grad(): pred = model(torch.from_numpy(d['states45'][:-1])).numpy()
            error = float(np.max(np.abs(pred-d['effective_actions11'])))
            replay_error = max(replay_error, error); assert error < 1e-4, error
            for k in CHECKERS:
                if not events[-1][k]: endings[k] += 1
        reader = imageio.get_reader(folder/'rollout.mp4'); count = sum(1 for _ in reader); reader.close()
        assert count == 201; video_counts.append({'folder': row['folder'], 'frames': count})
    for r, s in zip(dev, student):
        assert r['seed'] == s['seed']
        pair_audit=canonical_pair(ROOT/r['folder'],ROOT/s['folder'])
        save(OUT/f"initial_pair_{r['seed']}.json",pair_audit)
    counts = lambda rows: {k: sum(r[k] for r in rows) for k in ['ever_grasp', 'success_once', 'success_at_end']}
    summary = {'teacher_fit_all': counts(collection[:80]), 'teacher_dev': counts(dev), 'student_dev': counts(student),
               'fit_successful_demos': len(fit), 'fit_transitions': len(fit)*200,
               'gate_passed': complete['passed'], 'final_metrics': complete['final_supervised_metrics'],
               'selection': load(selection_path()), 'student_failed_end_conditions': dict(endings),
               'stages': {s: load((interface_path().parent if s == 'interface' else BASE)/(f'process_{s}_completed.json' if s=='train' and (BASE/'process_train_completed.json').exists() else f'process_{s}.json')) for s in ['witness', 'interface', 'collect', 'train']}}
    assert summary['student_dev'] == complete['counts']
    audit = {'passed': True, 'formal_episodes': len(all_rows), 'formal_steps': len(all_rows)*200,
             'video_counts': video_counts, 'student_action_replay_max_abs_error': replay_error,
             'train_only_normalization_verified': True, 'heldout_start_pairs_verified': 20,
             'updates_verified': len(updates), 'teacher_interface_pairs_verified': 20,
             'initial_pairing_scope':'Semantic init-config name; all active actors/robot/controller checked; only far hidden actor z parking slots canonicalized. Full teacher observations bit-identical.',
             'evaluation_recovery':'First failed episode retained; remaining 19 evaluated using unchanged update-5000 checkpoint; original failed process record preserved.'}
    save(OUT/'summary.json', summary); save(OUT/'audit.json', audit)
    metrics = load(BASE/'student/metrics.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for subset in ['fit', 'dev']:
        axes[0].plot([r['update'] for r in metrics], [r[subset]['total'] for r in metrics], label=subset)
    axes[0].set(xlabel='Gradient updates', ylabel='Motion MSE + gripper MSE', title='Supervised action loss'); axes[0].legend()
    keys = ['ever_grasp', 'success_once', 'success_at_end']; pos = np.arange(3)
    for offset, name in [(-.18, 'teacher_dev'), (.18, 'student_dev')]:
        axes[1].bar(pos+offset, [summary[name][k] for k in keys], .36, label=name)
    axes[1].set_xticks(pos, ['Ever grasp', 'Success once', 'Success at end']); axes[1].set_ylim(0, 20)
    axes[1].set(ylabel='Episodes / 20', title='Same-task held-out starts'); axes[1].legend()
    fig.tight_layout(); fig.savefig(OUT/'results.png', dpi=160); fig.savefig(OUT/'results.pdf'); plt.close(fig)
    # Illustrative videos only. Select first student success and first failure; never use for aggregate counts.
    selected = []
    for success in [True, False]:
        idx = next((i for i, r in enumerate(student) if r['success_once'] == success), None)
        if idx is not None: selected.append(idx)
    demo = OUT/'teacher_student_demo.mp4'
    writer = imageio.get_writer(demo, fps=20, codec='libx264', quality=7)
    try:
        for idx in selected:
            teacher_reader = imageio.get_reader(ROOT/dev[idx]['folder']/'rollout.mp4')
            student_reader = imageio.get_reader(ROOT/student[idx]['folder']/'rollout.mp4')
            for step, (left, right) in enumerate(zip(teacher_reader, student_reader)):
                canvas = Image.new('RGB', (1024, 576), '#121b2c'); draw = ImageDraw.Draw(canvas)
                draw.text((15, 8), f"Official SAC teacher | seed {dev[idx]['seed']}", fill='white')
                draw.text((527, 8), 'TRAINED BC STUDENT | 5000 updates', fill='white')
                draw.text((15, 34), f"Frame {step}/200 | Same train task, held-out start | Student success once: {student[idx]['success_once']}", fill='white')
                canvas.paste(Image.fromarray(left), (0, 64)); canvas.paste(Image.fromarray(right), (512, 64))
                writer.append_data(np.asarray(canvas))
            teacher_reader.close(); student_reader.close()
    finally: writer.close()
    save(OUT/'demo_selection.json', {'rule': 'first success-once and first no-success student episode in seed order',
                                   'indices': selected, 'seeds': [student[i]['seed'] for i in selected], 'frames': 201*len(selected)})
    print('BC_AUDIT_REPORT_PASS', json.dumps(summary), flush=True)


if __name__ == '__main__': main()
