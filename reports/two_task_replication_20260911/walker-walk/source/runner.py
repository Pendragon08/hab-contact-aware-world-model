"""Paper-budget DMControl experiment using the frozen official TD-MPC2 core."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
import time

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.buffer import Buffer
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

ROOT = Path(os.environ['DMC_ROOT'])
UPSTREAM = ROOT.parent / 'tdmpc2_original'
PLAN = json.loads((ROOT / 'configs/plan.json').read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    if isinstance(x, np.generic):
        return x.item()
    return x


def verify_sources():
    manifest = json.loads((UPSTREAM / 'upstream_manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((UPSTREAM / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    return manifest['commit']


def resource_guard(require_idle=False):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    raw = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    uuid, used, free = [x.strip() for x in raw.split(',')]
    assert int(free) >= 12288, raw
    assert shutil.disk_usage(ROOT).free >= 15 * 2**30, 'Disk reserve below 15 GiB'
    if require_idle:
        assert int(used) < 500, 'GPU0 occupied: ' + raw
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()), apps
    return {'gpu_uuid': uuid, 'used_MiB': int(used), 'free_MiB': int(free), 'disk_free_bytes': shutil.disk_usage(ROOT).free}


def configuration(seed, witness):
    c = OmegaConf.load(UPSTREAM / 'upstream/tdmpc2/config.yaml')
    del c['defaults']
    c.update(MODEL_SIZE[5])
    c.update({'task': PLAN['task'], 'obs': 'state', 'model_size': 5,
              'multitask': False, 'task_dim': 0, 'tasks': [PLAN['task']],
              'seed': seed, 'steps': 3000 if witness else PLAN['decisions_per_seed'],
              'eval_episodes': 1 if witness else PLAN['eval_episodes'],
              'eval_freq': PLAN['eval_frequency_decisions'], 'save_video': True,
              'bin_size': (c.vmax-c.vmin)/(c.num_bins-1)})
    return c


class RecordedEnv:
    def __init__(self, env, out):
        self.env, self.out = env, out
        self.phase = 'train'
        self.stage = 0
        self.counts = {'train': 0, 'eval': 0}
        self.episodes = {'train': 0, 'eval': 0}
        self.active = False
        self.eval_returns = []

    def rand_act(self):
        return self.env.rand_act()

    def reset(self):
        assert not self.active
        obs = self.env.reset()
        self.obs = [obs.numpy().copy()]
        self.actions, self.rewards = [], []
        self.physics = [self.env.unwrapped.physics.get_state().copy()]
        self.active = True
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward)
        assert torch.isfinite(action).all() and action.abs().max() <= 1.00001
        self.counts[self.phase] += 1
        self.obs.append(obs.numpy().copy())
        self.actions.append(action.numpy().copy())
        self.rewards.append(float(reward))
        self.physics.append(self.env.unwrapped.physics.get_state().copy())
        if done:
            assert len(self.actions) == 500
            folder = self.out / 'trajectories' / self.phase
            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{self.episodes[self.phase]:06d}'
            np.savez_compressed(folder / (stem + '.npz'), obs=np.asarray(self.obs),
                                action=np.asarray(self.actions), reward=np.asarray(self.rewards),
                                physics=np.asarray(self.physics), stage=np.array(self.stage))
            if self.phase == 'eval':
                self.eval_returns.append(sum(self.rewards))
            self.episodes[self.phase] += 1
            self.active = False
        return obs, reward, done, info

    def render(self):
        return self.env.render()


class Video:
    def __init__(self, out):
        self.out = out
        self.enabled = False

    def init(self, env, enabled=True):
        self.enabled = enabled
        self.frames = []
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render()
            assert frame.ndim == 3 and frame.shape[-1] == 3
            self.frames.append(frame)

    def save(self, step):
        if self.enabled:
            folder = self.out / 'videos'
            folder.mkdir(exist_ok=True)
            imageio.mimsave(folder / f'eval_{2*step:07d}.mp4', self.frames, fps=15)


class Logger:
    def __init__(self, out, env, agent):
        self.out, self.env, self.agent = out, env, agent
        self.video = Video(out)

    def log(self, d, category='train'):
        row = {k: scalar(v) for k, v in d.items()}
        # DMControl's default success=0 is a placeholder, not an outcome metric.
        row.pop('episode_success', None)
        row.update(category=category, train_decisions=self.env.counts['train'],
                   train_environment_steps=2*self.env.counts['train'],
                   eval_decisions=self.env.counts['eval'])
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        if category == 'eval':
            row['episode_returns'] = self.env.eval_returns[-self.agent.cfg.eval_episodes:]
        with (self.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row), flush=True)
        if category == 'eval':
            # Keep every evaluation trajectory/video; milestone weights limit disk use.
            if d['step'] % 500000 == 0 or d['step'] == self.agent.cfg.steps:
                folder = self.out / 'models'
                folder.mkdir(exist_ok=True)
                self.agent.save(folder / f'step_{2*d["step"]:07d}.pt')
        resource_guard()

    def finish(self, agent):
        agent.save(self.out / 'final.pt')


source = textwrap.dedent(inspect.getsource(OnlineTrainer.train))
assert source.count('while self._step <= self.cfg.steps:') == 1
source = source.replace('while self._step <= self.cfg.steps:', 'while self._step < self.cfg.steps:')
namespace = {}
exec(compile(source, 'official_train_exact_budget', 'exec'), OnlineTrainer.train.__globals__, namespace)


class Trainer(OnlineTrainer):
    train = namespace['train']

    def eval(self):
        # Same environment/RNG ordering as upstream. Only record the phase.
        self.env.phase = 'eval'
        self.env.stage = self._step
        try:
            return super().eval()
        finally:
            self.env.phase = 'train'


def hashes(agent):
    return {name: hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in module.parameters())).hexdigest()
            for name, module in {'encoder': agent.model._encoder, 'dynamics': agent.model._dynamics,
                                 'reward': agent.model._reward, 'Q': agent.model._Qs, 'policy': agent.model._pi}.items()}


def environment_witness(cfg):
    from dm_control import suite
    from envs.dmcontrol import make_env as make_dmc
    wrapped = make_dmc(cfg)
    raw = suite.load('walker', 'walk', task_kwargs={'random': cfg.seed}, visualize_reward=False)
    a = wrapped.reset()
    b = np.concatenate([v.flatten() for v in raw.reset().observation.values()])
    error = float(np.max(np.abs(a-b)))
    reward_error = 0.0
    for i in range(500):
        # Dyadic actions are represented exactly through official [-1,1] scaling.
        # Arbitrary float32 sin values change by ulps in (a+1)-1 and can diverge
        # through contact dynamics; that is not an action-repeat discrepancy.
        action = (((np.arange(6)+i) % 9 - 4) / 8).astype(np.float32)
        a, reward, done, _ = wrapped.step(action)
        r1, r2 = raw.step(action), raw.step(action)
        b = np.concatenate([v.flatten() for v in r2.observation.values()])
        error = max(error, float(np.max(np.abs(a-b))))
        reward_error = max(reward_error, abs(reward - (r1.reward + r2.reward)))
        assert done == (i == 499)
    assert error <= 1e-10 and reward_error <= 1e-10, (error, reward_error)
    assert wrapped.render().shape == (384, 384, 3)
    raw.close()
    wrapped.unwrapped.close()
    return {'decisions': 500, 'raw_environment_steps': 1000,
            'repeat_observation_max_error': error, 'repeat_reward_max_error': reward_error,
            'render_shape': [384, 384, 3]}


def main(mode, seed):
    preflight = resource_guard(require_idle=True)
    commit = verify_sources()
    witness = mode == 'witness'
    label = f'witness_{time.strftime("%Y%m%d_%H%M%S")}' if witness else f'walker_walk_seed{seed}_{PLAN["environment_steps_per_seed"]//1000000}m'
    out = ROOT / 'artifacts' / label
    out.mkdir(parents=True, exist_ok=False)
    files = [ROOT / 'runner.py', ROOT / 'runtime.sh', ROOT / 'run.sh', ROOT / 'configs/plan.json', ROOT / 'configs/env-spec.json']
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out / 'sources.json', frozen)
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'torchrl', 'tensordict', 'numpy', 'dm-control', 'mujoco', 'dm-env', 'dm-tree', 'gym', 'scipy']}
    save(out / 'runtime.json', versions)
    c = configuration(seed, witness)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.65)
    env_check = environment_witness(c) if witness else None
    set_seed(seed)
    env = make_env(c)
    assert tuple(c.obs_shape.state) == (24,) and c.action_dim == 6 and c.episode_length == 500
    assert c.seed_steps == 2500
    # Gym action-space RNG is otherwise independent of set_seed; explicitly seed it.
    env.action_space.seed(seed)
    env = RecordedEnv(env, out)
    agent = TDMPC2(c)
    before = hashes(agent)
    assert agent.discount == 0.99
    buffer = Buffer(c)
    logger = Logger(out, env, agent)
    trainer = Trainer(c, env, agent, buffer, logger)
    OmegaConf.save(c, out / 'resolved-config.yaml')
    started = time.time()
    update_count = 0
    original_update = agent.update

    def checked_update(replay):
        nonlocal update_count
        metrics = {k: scalar(v) for k, v in original_update(replay).items()}
        assert all(math.isfinite(v) for v in metrics.values()), metrics
        update_count += 1
        if update_count == 1 or update_count % 100 == 0:
            row = dict(update=update_count, train_decisions=env.counts['train'],
                       train_environment_steps=2*env.counts['train'], seconds=time.time()-started, **metrics)
            with (out / 'updates.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print('DMC_UPDATE', json.dumps(row), flush=True)
        return metrics

    agent.update = checked_update
    try:
        trainer.train()
        assert env.counts['train'] == c.steps and not env.active
        assert update_count == c.steps - 1
        final_eval = {k: scalar(v) for k, v in trainer.eval().items()}
        final_eval.update(trainer.common_metrics())
        logger.log(final_eval, 'eval')
        final_eval.pop('episode_success', None)
        after = hashes(agent)
        assert all(before[k] != after[k] for k in before)
        assert verify_sources() == commit
        for name, digest in frozen.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        result = {'completed': True, 'mode': mode, 'task': c.task, 'seed': seed,
                  'train_decisions': env.counts['train'], 'train_environment_steps': 2*env.counts['train'],
                  'eval_environment_steps': 2*env.counts['eval'], 'optimizer_updates': update_count,
                  'final_eval': final_eval, 'final_episode_returns': env.eval_returns[-c.eval_episodes:],
                  'all_five_modules_changed': True, 'seconds': time.time()-started,
                  'torch_peak_MiB': torch.cuda.max_memory_allocated()/2**20,
                  'preflight': preflight, 'environment_witness': env_check, 'upstream_commit': commit}
        save(out / 'complete.json', result)
        print('DMC_ORIGINAL_WITNESS_PASS' if witness else 'DMC_ORIGINAL_TRAINING_COMPLETE', json.dumps(result), flush=True)
    finally:
        env.env.unwrapped.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['witness', 'train'])
    parser.add_argument('--seed', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(args.mode, args.seed)
