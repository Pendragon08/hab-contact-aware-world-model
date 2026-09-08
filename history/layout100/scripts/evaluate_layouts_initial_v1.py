"""Frozen SAC development evaluation: explicit layouts, unmodified task criteria."""
import argparse, csv, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from demo import create_env, load_policy, frame
from mshab.utils.array import to_tensor

ROOT = Path(os.environ["HAB_ROOT"])
CHECKERS = ["is_grasped", "ee_rest", "robot_rest", "is_static", "cumulative_force_within_limit"]
FLOATS = ["robot_cumulative_force", "robot_force", "robot_target_pairwise_force"]

def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def classify(events):
    end = events[-1]
    ever = any(e["is_grasped"] for e in events)
    if end["success"]: category = "success"
    elif not ever: category = "never_grasped"
    elif not end["is_grasped"]: category = "not_holding_at_end"
    elif not end["cumulative_force_within_limit"]: category = "holding_force_limit"
    elif not (end["ee_rest"] and end["robot_rest"]): category = "holding_not_at_rest"
    elif not end["is_static"]: category = "holding_not_static"
    else: raise AssertionError("Unexplained failure despite all success checkers")
    stable_grasp, consecutive_grasp, consecutive_release, losses = False, 0, 0, []
    for e in events:
        if e["is_grasped"]:
            consecutive_grasp += 1; consecutive_release = 0
            if consecutive_grasp >= 3: stable_grasp = True
        else:
            consecutive_grasp = 0
            consecutive_release += 1
            if stable_grasp and consecutive_release == 3:
                losses.append(e["step"]-2)
                stable_grasp = False
    return {
        "outcome_category": category, "ever_grasped": ever,
        "grasp_loss_event_count": len(losses), "grasp_loss_start_steps": losses,
        "force_limit_exceeded_any": any(not e["cumulative_force_within_limit"] for e in events),
        "first_grasp_step": next((e["step"] for e in events if e["is_grasped"]), None),
        "first_success_step": next((e["step"] for e in events if e["success"]), None),
    }

def make_plan(env, policy_sha, path):
    if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(env.unwrapped.build_config_idx_to_task_plans)
    assert len(keys) >= 5, keys
    positions = np.linspace(0, len(keys)-1, 5).round().astype(int).tolist()
    names = {v:k for k,v in env.unwrapped.scene_builder.build_config_names_to_idxs.items()}
    slots = [{"slot":i, "build_config_idx":keys[p], "build_config_name":names[keys[p]],
              "episodes":20, "formal_seeds":list(range(20000+i*100, 20020+i*100)),
              "probe_seed":40000+i} for i,p in enumerate(positions)]
    plan = {
        "protocol":"HAB Pick cross-layout development evaluation v1",
        "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split":"val (development only; not final test)", "policy":"official SAC all-object",
        "checkpoint_sha256":policy_sha, "env_id":"PickSubtaskTrain-v0",
        "num_envs":1, "max_episode_steps":200, "formal_episodes":100,
        "layout_selection":"five equally spaced indices from sorted available validation build configs; fixed before rollouts",
        "available_build_config_idxs":keys, "layouts":slots,
        "object_selection":"original environment seeded task-plan and spawn sampler; not object-balanced",
        "success_definition":"unaltered conjunction of is_grasped, ee_rest, robot_rest, is_static, cumulative_force_within_limit",
        "terminal_success_source":"official info.episode.s_e; cross-checked against all five component checkers",
        "failure_priority":["success","never_grasped","not_holding_at_end","holding_force_limit","holding_not_at_rest","holding_not_static"],
        "failure_interpretation":"operational outcome categories, not proven causal explanations; other failed constraints retained as overlapping flags",
        "grasp_loss_definition":"three consecutive grasp=true control steps, followed by three consecutive false steps; a grasp-signal event, not proof of physical dropping",
        "primary_metrics":["success_once","success_at_end"],
        "example_selection":"lowest episode id for each observed outcome category; illustrative, not representative weighting",
        "hardware":"GPU1 only; GPU0 untouched",
        "source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/"scripts/evaluate_layouts.py", ROOT/"scripts/demo.py",
            ROOT/"third_party/mshab/mshab/envs/pick.py",
            ROOT/"third_party/mshab/mshab/envs/sequential_task.py"]},
    }
    save_json(path, plan)
    return plan

def rollout(env, policy, folder, slot, seed, episode_id, first):
    options = {"reconfigure":True, "build_config_idxs":[slot["build_config_idx"]]} if first else {}
    obs, info = env.reset(seed=seed, options=options)
    u = env.unwrapped
    assert list(u.build_config_idxs) == [slot["build_config_idx"]]
    actual = {"build_config_idxs":list(u.build_config_idxs), "build_config_name":slot["build_config_name"],
              "init_config_idxs":list(u.init_config_idxs),
              "task_plan_idxs":u.task_plan_idxs.detach().cpu().tolist(),
              "spawn_selection_idxs":list(u.spawn_selection_idxs),
              "target_object":u.subtask_objs[0]._objs[0].name}
    stem = f"episode_{episode_id:03d}"
    target = folder/stem
    target.mkdir(exist_ok=False)
    save_json(target/"scene.json", actual)
    torch.save({"environment_state":cpu_tree(u.get_state_dict()), "observation":cpu_tree(obs),
                "torch_rng":torch.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state_all(),
                "numpy_rng":np.random.get_state(), "python_rng":random.getstate(),
                "note":"audit snapshot; full paired restoration has not been validated for this evaluation"},
               target/"initial_snapshot.pt")
    writer = imageio.get_writer(target/"rollout.mp4", fps=20, codec="libx264", quality=7)
    events, actions, states, total_reward, times = [], [], [], 0.0, []
    started = time.perf_counter()
    writer.append_data(frame(env))
    ended = False
    try:
        with (target/"steps.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(1,201):
                obs = to_tensor(obs, device="cuda", dtype="float")
                states.append(obs["state"].detach().cpu().numpy()[0].copy())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    action = policy.actor(obs["pixels"], obs["state"], compute_pi=False, compute_log_pi=False)[0]
                torch.cuda.synchronize()
                policy_ms = (time.perf_counter()-t0)*1000
                assert torch.isfinite(action).all()
                actions.append(action.detach().cpu().numpy()[0].copy())
                obs, reward, term, trunc, info = env.step(action)
                ended = bool((term | trunc).any().item())
                current = info.get("final_info", info)
                success = bool(info["episode"]["s_e"][0].item()) if ended else bool(current["success"][0].item())
                event = {"step":step, "success":success, "reward":float(reward[0].item()),
                         "policy_ms":policy_ms}
                for key in CHECKERS:
                    assert key in current, key
                    event[key] = bool(current[key][0].item())
                for key in FLOATS:
                    if key in current: event[key] = float(current[key][0].item())
                assert success == all(event[key] for key in CHECKERS), (step,event)
                event["object_xyz"] = u.subtask_objs[0].pose.p[0].detach().cpu().tolist()
                event["tcp_xyz"] = u.agent.tcp_pose.p[0].detach().cpu().tolist()
                event["rest_xyz"] = u.ee_rest_world_pose.p[0].detach().cpu().tolist()
                events.append(event); times.append(policy_ms)
                total_reward += event["reward"]
                stream.write(json.dumps(event)+"\n")
                if step % 50 == 0: stream.flush()
                writer.append_data(frame(env))
                if ended:
                    terminal = {k:v[0].item() for k,v in info["episode"].items()}
                    save_json(target/"terminal_stats.json", terminal)
                    assert step == 200 and terminal["l"] == 200
                    assert bool(terminal["s_o"]) == any(e["success"] for e in events)
                    assert bool(terminal["s_e"]) == success
                    break
        assert ended and len(events) == 200
    finally:
        writer.close()
    np.savez_compressed(target/"trajectory.npz", policy_actions=np.stack(actions), observed_states=np.stack(states))
    row = {"episode_id":episode_id, "seed":seed, "layout_slot":slot["slot"],
           "build_config_idx":slot["build_config_idx"], "build_config_name":slot["build_config_name"],
           "target_object":actual["target_object"].split("env-0_")[-1],
           "steps":len(events), "success_once":any(e["success"] for e in events),
           "success_at_end":events[-1]["success"], "return":total_reward,
           "wall_seconds":time.perf_counter()-started,
           "policy_ms_mean":float(np.mean(times)), "policy_ms_p95":float(np.percentile(times,95)),
           "video":f"{stem}/rollout.mp4",
           "terminal_checkers":{k:events[-1][k] for k in CHECKERS}, **classify(events)}
    save_json(target/"episode.json", row)
    print("EPISODE_COMPLETE", json.dumps(row), flush=True)
    return row

def main(args):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    env = create_env()
    try:
        policy, policy_sha = load_policy(env)
        spec = json.loads((ROOT/"configs/env-spec.json").read_text())
        spec_hash = hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:8]
        assert spec_hash == "d873b21f", spec_hash
        plan_path = ROOT/"configs/hab100_v1.json"
        if args.mode == "run": assert plan_path.exists()
        plan = make_plan(env, policy_sha, plan_path)
        assert plan["checkpoint_sha256"] == policy_sha
        for name, expected in plan["source_hashes"].items(): assert sha(ROOT/name) == expected, name
        chosen = plan["layouts"] if args.mode == "probe" else [plan["layouts"][args.layout_slot]]
        folder = ROOT/args.output
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for slot in chosen:
            seeds = [slot["probe_seed"]] if args.mode == "probe" else slot["formal_seeds"]
            for j,seed in enumerate(seeds):
                episode_id = slot["slot"]*20+j
                rows.append(rollout(env,policy,folder,slot,seed,episode_id,j==0))
                save_json(folder/f"slot_{slot['slot']}_progress.json", rows)
        save_json(folder/f"complete_{args.mode}_{args.layout_slot}.json",
                  {"mode":args.mode,"episodes":len(rows),"interaction_steps":200*len(rows),
                   "spec_hash":spec_hash,"checkpoint_sha256":policy_sha,
                   "torch_peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2})
        print("JOB_COMPLETE", args.mode, args.layout_slot, len(rows), flush=True)
    finally:
        env.close()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["probe","run"],required=True)
    p.add_argument("--layout-slot",type=int,default=0)
    p.add_argument("--output",required=True)
    main(p.parse_args())

