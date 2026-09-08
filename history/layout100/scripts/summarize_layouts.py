"""Audit complete rollout artifacts and build descriptive development metrics."""
import argparse, csv, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

LABELS = {
 "success":"结束时成功", "never_grasped":"全程未出现抓持",
 "not_holding_at_end":"曾抓持但结束未抓持", "holding_force_limit":"仍抓持但累计力超限",
 "holding_not_at_rest":"仍抓持但未满足回收姿态", "holding_not_static":"仍抓持且姿态达标但不稳定"}
KEYS = ["is_grasped","ee_rest","robot_rest","is_static","cumulative_force_within_limit"]

def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def interval(k,n):
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-delta, center+delta]

def export_csv(path,rows):
    flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows]
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)

def group_stats(rows):
    n=len(rows); once=sum(r["success_once"] for r in rows); end=sum(r["success_at_end"] for r in rows)
    first=[r["first_success_step"] for r in rows if r["first_success_step"] is not None]
    return {"episodes":n,"success_once_count":once,"success_at_end_count":end,
            "success_once_rate":once/n,"success_at_end_rate":end/n,
            "once_wilson95":interval(once,n),"end_wilson95":interval(end,n),
            "mean_first_success_step_conditional":float(np.mean(first)) if first else None,
            "ever_grasped_count":sum(r["ever_grasped"] for r in rows),
            "grasp_loss_event_episode_count":sum(r["grasp_loss_event_count"]>0 for r in rows),
            "force_limit_exceeded_episode_count":sum(r["force_limit_exceeded_any"] for r in rows),
            "mean_episode_seconds":float(np.mean([r["wall_seconds"] for r in rows])),
            "outcome_counts":dict(Counter(r["outcome_category"] for r in rows))}

def summarize(root):
    folder=root/"results/pick_val_layout100_v1"
    plan=json.loads((root/"configs/hab100_v1.json").read_text(encoding="utf-8"))
    expected={slot["slot"]*20+j:(slot,seed) for slot in plan["layouts"] for j,seed in enumerate(slot["formal_seeds"])}
    rows=[]; audits=[]; policy_times=[]; all_steps=[]; overlaps=Counter()
    for ep_id,(slot,seed) in sorted(expected.items()):
        sub=folder/f"episode_{ep_id:03d}"
        row=json.loads((sub/"episode.json").read_text(encoding="utf-8"))
        row["object_category"]=re.sub(r"-\d+$","",row["target_object"])
        events=[json.loads(x) for x in (sub/"steps.jsonl").read_text().splitlines()]
        row["first_force_limit_step"]=next((e["step"] for e in events if not e["cumulative_force_within_limit"]),None)
        terminal=json.loads((sub/"terminal_stats.json").read_text())
        scene=json.loads((sub/"scene.json").read_text())
        assert row["episode_id"]==ep_id and row["seed"]==seed
        assert row["build_config_idx"]==slot["build_config_idx"]==scene["build_config_idxs"][0]
        assert row["steps"]==len(events)==terminal["l"]==200
        assert row["success_once"]==bool(terminal["s_o"])==any(e["success"] for e in events)
        assert row["success_at_end"]==bool(terminal["s_e"])==events[-1]["success"]
        for i,e in enumerate(events,1):
            assert e["step"]==i
            assert e["success"]==all(e[k] for k in KEYS)
            assert all(np.isfinite(e[k]) for k in ["reward","policy_ms"])
            policy_times.append(e["policy_ms"])
            all_steps.append({"episode_id":ep_id,"layout_slot":slot["slot"], **e})
        assert abs(sum(e["reward"] for e in events)-row["return"])<1e-4
        traj=np.load(sub/"trajectory.npz")
        assert len(traj["policy_actions"])==len(traj["observed_states"])==200
        assert np.isfinite(traj["policy_actions"]).all() and np.isfinite(traj["observed_states"]).all()
        if not row["success_at_end"]:
            for k in KEYS:
                if not events[-1][k]: overlaps[k]+=1
        rows.append(row)
        audits.append({"episode_id":ep_id,"steps":len(events),"terminal_success_checks":"PASS","layout":"PASS","trajectory":"PASS"})
    assert len(rows)==100 and len({r["build_config_idx"] for r in rows})==5
    by_layout=[]
    for slot in plan["layouts"]:
        group=[r for r in rows if r["layout_slot"]==slot["slot"]]
        assert len(group)==20
        by_layout.append({"layout_slot":slot["slot"],"build_config_idx":slot["build_config_idx"],
                          "build_config_name":slot["build_config_name"], **group_stats(group)})
    by_object=[]
    for obj in sorted({r["object_category"] for r in rows}):
        by_object.append({"object":obj,**group_stats([r for r in rows if r["object_category"]==obj])})
    examples=[]
    for cat in LABELS:
        matching=[r for r in rows if r["outcome_category"]==cat]
        if matching: examples.append({"category":cat,"episode_id":min(r["episode_id"] for r in matching)})
    summary={"scope":plan["split"],"policy":plan["policy"],"checkpoint_sha256":plan["checkpoint_sha256"],
             "formal_interaction_steps":20000,"probe_interaction_steps":3000,
             "layout_count":5, **group_stats(rows),
             "policy_latency_ms":{"mean":float(np.mean(policy_times)),"p50":float(np.percentile(policy_times,50)),"p95":float(np.percentile(policy_times,95))},
             "failed_episode_terminal_conditions_overlapping":dict(overlaps),"examples":examples,
             "once_then_end_failure_count":sum(r["success_once"] and not r["success_at_end"] for r in rows),
             "once_then_end_failure_categories":dict(Counter(r["outcome_category"] for r in rows if r["success_once"] and not r["success_at_end"])),
             "uncertainty_note":"Wilson intervals are descriptive binomial intervals conditional on this checkpoint and selected layouts; not training-seed variability or guaranteed cross-scene generalization.",
             "category_note":"priority-based observed outcome categories, not causal attribution; grasp-signal loss is not proof of dropping.",
             "all_episode_audits_passed":True}
    for filename,data in [("summary.json",summary),("episodes.json",rows),("by_layout.json",by_layout),("by_object.json",by_object),("audit.json",audits)]:
        write(folder/filename,data)
    export_csv(folder/"episodes.csv",rows); export_csv(folder/"steps.csv",all_steps)
    export_csv(folder/"by_layout.csv",by_layout); export_csv(folder/"by_object.csv",by_object)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True)
    summarize(Path(p.parse_args().root))
