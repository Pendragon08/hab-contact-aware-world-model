"""Compare saved model predictions with a causal last-reward reference."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/world_model_prediction"


def errors(predicted, truth):
    error = predicted - truth
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def main():
    original = json.loads((DEST / "summary.json").read_text(encoding="utf-8"))
    reference = original["constant_reward"]
    results = {}
    for behavior in ["tdmpc2", "sac"]:
        folder = ROOT / "artifacts" / f"state_{behavior}_pilot_v1"
        rewards = []
        for episode in range(20):
            record = json.loads((folder / f"eval_{episode:03d}.json").read_text(encoding="utf-8"))
            assert record["split"] == "val-development" and len(record["steps"]) == 200
            rewards.append([r["reward"] for r in record["steps"]])
        rewards = np.asarray(rewards, dtype=np.float32)
        with np.load(DEST / f"{behavior}_prediction_data.npz") as data:
            ids = data["episode_and_start"]
            keep = ids[:, 1] >= 1  # The first state has no previously observed reward.
            episode, start = ids[keep].T
            truth = data["reward_truth"][keep]
            model = data["reward_prediction"][keep]
            expected = rewards[episode[:, None], start[:, None] + np.arange(3)[None, :]]
            np.testing.assert_array_equal(truth, expected)
            past = rewards[episode, start - 1][:, None]
            assert truth.shape == model.shape == (3940, 3)
            assert np.isfinite(model).all()
        table = []
        for horizon in range(3):
            target = truth[:, horizon]
            table.append({
                "horizon": horizon + 1,
                "windows": len(target),
                "model": errors(model[:, horizon], target),
                "train_mean": errors(reference, target),
                "last_observed_reward": errors(past[:, 0], target),
            })
        results[behavior] = table
    audit = {
        "passed": True,
        "scope": "post-hoc causal persistence reference on development failure trajectories; no retraining or evaluation-data fitting",
        "window_selection": "starts 1..197 of every 200-step evaluation episode; all references use the same 3940 windows per behavior",
        "persistence_definition": "repeat reward at action index start-1 for all future horizons; no future reward is used as input",
        "caveat": "overlapping windows are correlated; both evaluated policies have 0/20 task success",
        "metrics_by_behavior": results,
    }
    (DEST / "persistence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# 补充：与上一时刻奖励保持不变的参考比较", "",
        "训练奖励均值是较弱的时间序列参考。因此补充一个无需训练、符合因果顺序的参考：预测时只读取上一步已获得的奖励，并把它保持到未来 1/2/3 步。该诊断不训练模型，也不利用验证奖励拟合参数。", "",
        "第一个状态没有上一奖励，故排除每回合起点 0。每种行为的全部方法都比较相同的 3,940 个窗口（20 回合 × 197），不与原报告 3,960 个窗口的数值直接混算。原报告和原始数据保持不变。", "",
        "|行为数据|步长|模型 MAE|训练均值 MAE|上一奖励 MAE|模型 RMSE|上一奖励 RMSE|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for behavior, table in results.items():
        for row in table:
            lines.append(f"|{behavior}|{row['horizon']}|{row['model']['mae']:.6f}|{row['train_mean']['mae']:.6f}|{row['last_observed_reward']['mae']:.6f}|{row['model']['rmse']:.6f}|{row['last_observed_reward']['rmse']:.6f}|")
    lines += [
        "", "两种学习策略在本轮均为 0/20 完整成功，因此本诊断主要覆盖失败轨迹中的密集奖励变化，不能代表抓取接触突变或成功轨迹预测。相邻窗口重叠，不作为独立样本做显著性结论。",
        "", "如果简单的上一奖励参考已经足够准确，世界模型优于训练均值仍不足以证明它学到了有助于控制的动力学。后续应增加任务成功与接触变化覆盖，评估动作相关的物理量预测、校准和实际闭环任务表现。",
        "", "完整数值和窗口口径见 `persistence_audit.json`，计算入口为 `scripts/audit_reward_persistence.py`。", "",
    ]
    (DEST / "时间序列参考诊断.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
