"""Describe completed validation failures without treating conditions as causes."""
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CHECKERS = ["is_grasped", "ee_rest", "robot_rest", "is_static", "cumulative_force_within_limit"]
LABELS = {
    "is_grasped": "抓住目标",
    "ee_rest": "末端到达休止位置",
    "robot_rest": "机器人到达休止状态",
    "is_static": "静止",
    "cumulative_force_within_limit": "累计接触未超限",
}


def main():
    dest = ROOT / "reports/state_pilot"
    audited = json.loads((dest / "audit.json").read_text(encoding="utf-8"))
    rows, summary = [], {}
    for algorithm in ["tdmpc2", "sac"]:
        assert audited["algorithms"][algorithm]["passed"]
        folder = ROOT / "artifacts" / f"state_{algorithm}_pilot_v1"
        records = []
        for episode in range(20):
            data = json.loads((folder / f"eval_{episode:03d}.json").read_text(encoding="utf-8"))
            steps = data["steps"]
            assert data["split"] == "val-development" and len(steps) == 200
            assert all(bool(e["success"]) == all(e[k] for k in CHECKERS) for e in steps)
            row = {
                "algorithm": algorithm,
                **data["summary"],
                "ever_grasped": any(e["is_grasped"] for e in steps),
                "return": sum(e["reward"] for e in steps),
                "final_cumulative_force": steps[-1]["robot_cumulative_force"],
                **{f"final_{k}": steps[-1][k] for k in CHECKERS},
            }
            records.append(row)
            rows.append(row)
        summary[algorithm] = {
            "episodes": len(records),
            "ever_grasped_count": sum(r["ever_grasped"] for r in records),
            "success_once_count": sum(r["success_once"] for r in records),
            "success_at_end_count": sum(r["success_at_end"] for r in records),
            "mean_return": mean(r["return"] for r in records),
            "final_condition_pass_counts": {
                k: sum(r[f"final_{k}"] for r in records) for k in CHECKERS
            },
        }
    result = {
        "summary": summary,
        "interpretation": "Descriptive validation conditions; overlapping conditions, not causal failure attribution or final-test evidence.",
    }
    (dest / "failure_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (dest / "validation_conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 状态策略验证条件诊断", "",
        "逐回合核对两算法各 20 个已完成验证回合。以下是观测到的条件，不是失败原因的因果归因；多个条件可能同时不满足。", "",
        "|算法|至少一次抓住目标|曾完整成功|结束完整成功|平均回合奖励|",
        "|---|---:|---:|---:|---:|",
    ]
    for algorithm, data in summary.items():
        lines.append(f"|{algorithm}|{data['ever_grasped_count']}/20|{data['success_once_count']}/20|{data['success_at_end_count']}/20|{data['mean_return']:.4f}|")
    lines += ["", "|第 200 步满足的条件|TD-MPC2 /20|状态 SAC /20|", "|---|---:|---:|"]
    for key in CHECKERS:
        lines.append(f"|{LABELS[key]}|{summary['tdmpc2']['final_condition_pass_counts'][key]}|{summary['sac']['final_condition_pass_counts'][key]}|")
    lines += [
        "", "抓住物体与完整完成 HAB Pick 是不同指标：还需要休止姿态、静止与接触约束同时成立。奖励也不能替代成功判据。",
        "", "下一阶段应先确认策略能够稳定接近和抓住目标，再评估完整技能完成。状态观测缺少完整家具和障碍几何，有限训练预算也可能影响结果；本轮没有分离这些因素，因此不能将零成功归因于某一个原因。",
        "", "随后需要固定新的开发协议，扩大训练并匹配基线观测、预算和评估条件。只有任务基线具备可用成功率后，才适合检验学习到的风险预测与规划是否带来增益。",
        "", "逐回合证据见 `validation_conditions.csv`，汇总见 `failure_diagnostics.json`；全部原始逐步事件保存在 Release 中。", "",
    ]
    (dest / "验证条件诊断.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
