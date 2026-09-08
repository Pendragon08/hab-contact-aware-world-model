"""Small, seeded inference demo, using the official HAB SAC actor and wrappers."""
import argparse, csv, hashlib, json, os, random, time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf
import mani_skill
from mani_skill import ASSET_DIR
from mshab.envs.make import EnvConfig, make_env
from mshab.agents.sac import Agent as SACAgent
from mshab.utils.array import to_tensor

def create_env():
    folder = ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange'
    cfg = EnvConfig(
        env_id='PickSubtaskTrain-v0', num_envs=1, max_episode_steps=200,
        obs_mode='depth', render_mode='rgb_array', sim_backend='gpu',
        record_video=False, frame_stack=3, continuous_task=True,
        task_plan_fp=str(folder/'task_plans/tidy_house/pick/val/all.json'),
        spawn_data_fp=str(folder/'spawn_data/tidy_house/pick/val/spawn_data.pt'),
        env_kwargs={'require_build_configs_repeated_equally_across_envs':False,
                    'add_event_tracker_info':True},
    )
    env = make_env(cfg)
    # Manual reset makes the terminal video frame correspond to terminal metrics.
    env.env.auto_reset = False
    return env

def load_policy(env):
    ckpt_dir = ASSET_DIR / 'mshab_checkpoints/rl/tidy_house/pick/all'
    algo = OmegaConf.load(ckpt_dir/'config.yml').algo
    pixel_spaces = {}
    for k, box in env.single_observation_space['pixels'].items():
        shape, lo, hi = box.shape, box.low, box.high
        if len(shape) == 4:
            shape = (shape[0]*shape[1], shape[-2], shape[-1])
            lo, hi = lo.reshape(shape), hi.reshape(shape)
        pixel_spaces[k] = spaces.Box(lo, hi, shape, box.dtype)
    keys = ['actor_hidden_dims','critic_hidden_dims','critic_layer_norm','critic_dropout',
            'encoder_pixels_feature_dim','encoder_state_feature_dim','cnn_features',
            'cnn_filters','cnn_strides','cnn_padding']
    kwargs = {k:OmegaConf.to_container(algo[k]) if OmegaConf.is_config(algo[k]) else algo[k] for k in keys}
    policy = SACAgent(spaces.Dict(pixel_spaces), env.single_observation_space['state'].shape,
                      env.single_action_space.shape, **kwargs,
                      log_std_min=algo.actor_log_std_min, log_std_max=algo.actor_log_std_max,
                      device='cuda')
    path = ckpt_dir/'policy.pt'
    policy.load_state_dict(torch.load(path, map_location='cuda', weights_only=False)['agent'], strict=True)
    policy.eval()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print('POLICY_LOADED', path, digest, flush=True)
    return policy, digest

def frame(env):
    img = env.unwrapped.render()
    if isinstance(img, torch.Tensor): img = img.detach().cpu().numpy()
    if img.ndim == 4: img = img[0]
    assert img.ndim == 3 and img.shape[-1] == 3, img.shape
    return img.astype(np.uint8)

def run(args):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    x = torch.randn(16, 16, device='cuda')
    y = x @ x.T
    assert torch.isfinite(y).all() and y.abs().sum() > 0
    print('WITNESS CUDA', torch.cuda.get_device_name(), tuple(y.shape), float(y.sum()), flush=True)
    print('MANISKILL_SOURCE', mani_skill.__file__, 'ASSET_DIR', ASSET_DIR, flush=True)
    env = create_env()
    policy, sha = load_policy(env)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    begin = time.time()
    try:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            obs, info = env.reset(seed=seed)
            scene_meta = {'build_config_idxs':list(env.unwrapped.build_config_idxs),
                          'init_config_idxs':list(env.unwrapped.init_config_idxs),
                          'spawn_selection_idxs':list(env.unwrapped.spawn_selection_idxs),
                          'target_object':env.unwrapped.subtask_objs[0]._objs[0].name}
            print('SCENE', json.dumps(scene_meta), flush=True)
            writer = None if args.validate else imageio.get_writer(out/f'episode_seed_{seed:03d}.mp4', fps=20, codec='libx264', quality=8)
            any_success, final_success, first_success, total_reward = False, False, None, 0.0
            actions, per_step = [], []
            initial_frame = frame(env)
            if writer: writer.append_data(initial_frame)
            try:
                for step in range(2 if args.validate else 200):
                    obs = to_tensor(obs, device='cuda', dtype='float')
                    with torch.no_grad():
                        action = policy.actor(obs['pixels'], obs['state'], compute_pi=False, compute_log_pi=False)[0]
                    assert torch.isfinite(action).all()
                    obs, reward, term, trunc, info = env.step(action)
                    current = info.get('final_info', info)
                    ended = bool((term | trunc).any())
                    # The official statistics wrapper resets an aliased info['success']
                    # tensor on done. Its copied episode['s_e'] is the terminal metric.
                    final_success = bool(info['episode']['s_e'][0].item()) if ended else bool(current['success'][0].item())
                    any_success |= final_success
                    if final_success and first_success is None: first_success = step+1
                    total_reward += float(reward[0])
                    actions.append(action.cpu().numpy()[0])
                    event = {'step':step+1, 'success':final_success, 'reward':float(reward[0])}
                    if ended:
                        terminal_stats = {k:v[0].item() for k,v in info['episode'].items()}
                        (out/f'terminal_stats_seed_{seed:03d}.json').write_text(json.dumps(terminal_stats,indent=2))
                        event['success_source'] = 'official episode.s_e'
                    for key in ['is_grasped','ee_rest','robot_rest','is_static','cumulative_force_within_limit','robot_cumulative_force']:
                        if key in current and isinstance(current[key], torch.Tensor): event[key] = current[key][0].item()
                    per_step.append(event)
                    if writer: writer.append_data(frame(env))
                    if step % 50 == 0: print('PROGRESS', seed, step+1, 'success', final_success, flush=True)
                    if ended: break
            finally:
                if writer: writer.close()
            row = {'seed':seed, 'steps':step+1, 'success_once':any_success,
                   'success_at_end':final_success, 'first_success_step':first_success,
                   'return':total_reward, 'video':None if args.validate else f'episode_seed_{seed:03d}.mp4'}
            rows.append(row)
            np.save(out/f'actions_seed_{seed:03d}.npy', np.stack(actions))
            (out/f'events_seed_{seed:03d}.json').write_text(json.dumps(per_step, indent=2))
            (out/f'scene_seed_{seed:03d}.json').write_text(json.dumps(scene_meta, indent=2))
            (out/'episodes.json').write_text(json.dumps(rows, indent=2))
            print('EPISODE', json.dumps(row), flush=True)
            if args.validate:
                imageio.imwrite(out/'witness.png', frame(env))
                print('SIM_WITNESS', initial_frame.shape, 'finite action; real GPU sim and render; checkpoint strict load', flush=True)
        with (out/'episodes.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        summary = {'task':'TidyHouse Pick subtask', 'env_id':'PickSubtaskTrain-v0', 'split':'val',
                   'policy':'official pretrained SAC all-object', 'checkpoint_sha256':sha,
                   'episodes':len(rows), 'success_once_count':sum(r['success_once'] for r in rows),
                   'success_at_end_count':sum(r['success_at_end'] for r in rows),
                   'wall_seconds':time.time()-begin,
                   'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                   'validation_only':args.validate}
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        print('COMPLETE', json.dumps(summary), flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--output', default='results/pick_val_seed0_4')
    p.add_argument('--validate', action='store_true')
    run(p.parse_args())
