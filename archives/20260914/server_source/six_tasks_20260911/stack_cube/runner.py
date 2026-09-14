"""Original ManiSkill2 Stack Cube, 1M-budget TD-MPC2 reproduction."""
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
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import numpy as np
import torch
import gym
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.buffer import Buffer
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer

ROOT = Path(os.environ['MS_ROOT'])
UPSTREAM = Path('/data/code/ouyanghao/oyh/hab-contact-aware-world-model/tdmpc2_original')
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
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    raw = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    uuid, used, free = [x.strip() for x in raw.split(',')]
    assert int(free) >= 12288, raw
    gpu0_free = int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    assert gpu0_free >= 12288, gpu0_free
    assert shutil.disk_usage(ROOT).free >= 15 * 2**30, 'Disk reserve below 15 GiB'
    if require_idle:
        assert int(used) < 500, 'GPU1 occupied: ' + raw
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        assert not any(line.split(',')[0].strip() == uuid for line in apps.splitlines()), apps
    return {'gpu_uuid': uuid, 'used_MiB': int(used), 'free_MiB': int(free), 'disk_free_bytes': shutil.disk_usage(ROOT).free}


def configuration(seed, witness):
    c = OmegaConf.load(UPSTREAM / 'upstream/tdmpc2/config.yaml')
    del c['defaults']
    c.update(MODEL_SIZE[5])
    c.update({'task': PLAN['task'], 'obs': 'state', 'model_size': 5,
              'multitask': False, 'task_dim': 0, 'tasks': [PLAN['task']],
              'seed': seed, 'steps': 1200 if witness else PLAN['decisions_per_seed'],
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
        self.eval_success_once = []

    def rand_act(self):
        return self.env.rand_act()

    def reset(self):
        assert not self.active
        obs = self.env.reset()
        self.obs = [obs.numpy().copy()]
        self.actions, self.rewards = [], []
        self.physics = [self.env.unwrapped.get_state().copy()]
        self.successes = []
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
        self.physics.append(self.env.unwrapped.get_state().copy())
        self.successes.append(bool(info['success']))
        if done:
            assert len(self.actions) == 100
            folder = self.out / 'trajectories' / self.phase
            folder.mkdir(parents=True, exist_ok=True)
            stem = f'{self.episodes[self.phase]:06d}'
            np.savez_compressed(folder / (stem + '.npz'), obs=np.asarray(self.obs),
                                action=np.asarray(self.actions), reward=np.asarray(self.rewards),
                                physics=np.asarray(self.physics), success=np.asarray(self.successes), stage=np.array(self.stage))
            if self.phase == 'eval':
                self.eval_returns.append(sum(self.rewards))
                self.eval_success_once.append(any(self.successes))
            self.episodes[self.phase] += 1
            self.active = False
        return obs, reward, done, info

    def render(self):
        return self.env.unwrapped.render(mode='cameras')


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
        row.update(category=category, train_decisions=self.env.counts['train'],
                   train_environment_steps=2*self.env.counts['train'],
                   eval_decisions=self.env.counts['eval'])
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
        if category == 'eval':
            row['episode_returns'] = self.env.eval_returns[-self.agent.cfg.eval_episodes:]
            row['success_once'] = float(np.mean(self.env.eval_success_once[-self.agent.cfg.eval_episodes:]))
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


def make_environment(cfg):
    # Inject only GPU selection into the untouched official environment constructor.
    original_make = gym.make
    def bound_make(name, *args, **kwargs):
        if name == PLAN['env_id']:
            kwargs['renderer_kwargs'] = dict(offscreen_only=True, device=os.environ['MS_RENDER_DEVICE'])
        return original_make(name, *args, **kwargs)
    gym.make = bound_make
    try:
        return make_env(cfg)
    finally:
        gym.make = original_make


def gpu_binding_evidence():
    gpu_xml = ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
    records = []
    for g in gpu_xml.findall('gpu'):
        for p in g.findall('./processes/process_info'):
            if int(p.findtext('pid')) == os.getpid():
                records.append({'gpu':int(g.findtext('minor_number')), 'type':p.findtext('type'),
                                'memory_MiB':int(p.findtext('used_memory').split()[0])})
    assert any(r['gpu']==1 and 'C' in r['type'] for r in records), records
    # Vulkan enumerates GPU0 and retains a 4 MiB graphics context even with pci:65.
    # Permit only this measured context; no compute or material rendering allocation.
    assert all(r['gpu']==1 or (r['gpu']==0 and r['type']=='G' and r['memory_MiB']<=8)
               for r in records), records
    return records


def environment_witness(cfg):
    env = make_environment(cfg)
    traces = []
    for repeat in range(2):
        env.unwrapped.seed(731)
        obs = env.reset()
        assert obs.shape == (55,)
        trace = [obs.numpy().copy()]
        for i in range(100):
            action = torch.tensor([.1, -.1, .1, 1.], dtype=torch.float32)
            obs, reward, done, info = env.step(action)
            assert done == (i == 99)
            assert env.unwrapped.elapsed_steps == 2*(i+1)
            assert bool(info['success']) == bool(env.unwrapped.evaluate()['success'])
            trace.append(np.concatenate([obs.numpy(), [float(reward), float(info['success'])]]))
        traces.append(trace)
    error = max(float(np.max(np.abs(a-b))) for a,b in zip(*traces))
    assert error <= 1e-6, error
    frame = env.unwrapped.render(mode='cameras')
    assert frame.ndim == 3 and frame.shape[-1] == 3 and np.ptp(frame)>0
    gpu_evidence = gpu_binding_evidence()
    native = env.unwrapped
    # Verify the native predicate can distinguish a known goal state from failure.
    # This synthetic check is separate from all policy trajectories and scores.
    from sapien.core import Pose
    # Reset contact records before constructing a separate synthetic state.
    env.reset()
    initial = native.evaluate()
    native.cubeB.set_pose(Pose([0.15, 0.15, 0.02]))
    native.cubeA.set_pose(Pose([0.15, 0.15, 0.06]))
    native.cubeA.set_velocity(np.zeros(3))
    native.cubeA.set_angular_velocity(np.zeros(3))
    native.agent.robot.set_qvel(np.zeros_like(native.agent.robot.get_qvel()))
    positive = bool(native.evaluate()['success'])
    assert positive
    native.cubeA.set_pose(Pose([-0.15, -0.15, 0.02]))
    negative = bool(native.evaluate()['success'])
    assert not negative
    check = {'native_info_keys': list(initial), 'reward_mode': native.reward_mode,
             'control_mode': native.control_mode, 'renderer_device_requested': os.environ['MS_RENDER_DEVICE'],
             'synthetic_success_predicate_positive':positive,'synthetic_success_predicate_negative':negative,
             'gpu_binding_evidence':gpu_evidence, 'physical_compute_gpu':1}
    env.close()
    return {'decisions_per_episode':100,'raw_steps_per_episode':200,'repeat_error':error,'render_shape':list(frame.shape),**check}


def main(mode, seed):
    preflight = resource_guard(require_idle=True)
    commit = verify_sources()
    witness = mode == 'witness'
    label = f'witness_{time.strftime("%Y%m%d_%H%M%S")}' if witness else f'stack_cube_seed{seed}_1m'
    out = ROOT / 'artifacts' / label
    out.mkdir(parents=True, exist_ok=False)
    files = [ROOT / 'runner.py', ROOT / 'runtime.sh', ROOT / 'run.sh', ROOT / 'configs/plan.json', ROOT / 'configs/env-spec.json']
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out / 'sources.json', frozen)
    versions = {n: importlib.metadata.version(n) for n in ['torch', 'torchrl', 'tensordict', 'numpy', 'mani-skill2', 'sapien', 'gym', 'scipy']}
    save(out / 'runtime.json', versions)
    c = configuration(seed, witness)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.65)
    env_check = environment_witness(c) if witness else None
    set_seed(seed)
    env = make_environment(c)
    assert tuple(c.obs_shape.state) == (55,) and c.action_dim == 4 and c.episode_length == 100
    assert c.seed_steps == 1000
    # Gym action-space RNG is otherwise independent of set_seed; explicitly seed it.
    env.action_space.seed(seed)
    env = RecordedEnv(env, out)
    agent = TDMPC2(c)
    before = hashes(agent)
    assert agent.discount == 0.95
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
            print('MS2_UPDATE', json.dumps(row), flush=True)
        return metrics

    agent.update = checked_update
    try:
        trainer.train()
        assert env.counts['train'] == c.steps and not env.active
        assert update_count == c.steps - 1
        final_eval = {k: scalar(v) for k, v in trainer.eval().items()}
        final_eval.update(trainer.common_metrics())
        logger.log(final_eval, 'eval')
        after = hashes(agent)
        assert all(before[k] != after[k] for k in before)
        assert verify_sources() == commit
        for name, digest in frozen.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        result = {'completed': True, 'mode': mode, 'task': c.task, 'seed': seed,
                  'train_decisions': env.counts['train'], 'train_environment_steps': 2*env.counts['train'],
                  'eval_environment_steps': 2*env.counts['eval'], 'optimizer_updates': update_count,
                  'final_eval': final_eval, 'final_episode_returns': env.eval_returns[-c.eval_episodes:],
                  'final_success_once': float(np.mean(env.eval_success_once[-c.eval_episodes:])),
                  'all_five_modules_changed': True, 'seconds': time.time()-started,
                  'torch_peak_MiB': torch.cuda.max_memory_allocated()/2**20,
                  'preflight': preflight, 'environment_witness': env_check, 'upstream_commit': commit}
        result['gpu_binding_evidence'] = gpu_binding_evidence()
        save(out / 'complete.json', result)
        print('MS2_ORIGINAL_WITNESS_PASS' if witness else 'MS2_ORIGINAL_TRAINING_COMPLETE', json.dumps(result), flush=True)
    finally:
        env.env.unwrapped.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['witness', 'train'])
    parser.add_argument('--seed', type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(args.mode, args.seed)
