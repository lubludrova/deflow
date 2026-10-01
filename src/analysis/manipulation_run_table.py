"""Summarize manipulation returns and success across supplied evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

from return_metrics import mean_ci

ROOT = Path(__file__).resolve().parents[2]
TASKS = [("ms_pickcube", "PickCube"), ("ms_pushcube", "PushCube"),
         ("mw_button", "ButtonPressWall"), ("mw_peg_native", "PegInsertSide"),
         ("mw_pushwall", "PushWall")]
METHODS = [("sac", "SAC"), ("sacflow", "SAC-Flow"), ("dime", "DIME"), ("deflow", "DEFlow")]


def table_cell(values):
    summary = mean_ci(values)
    half_width = (summary["ci95"][1] - summary["ci95"][0]) / 2
    return rf'${summary["mean"]:.1f} \pm {half_width:.1f}$ ({summary["n"]})'


def render(records):
    assert len({r["run_id"] for r in records}) == len(records)
    lines = [r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{6pt}",
             r"\caption{Manipulation benchmark results, mean $\pm$ 95\% confidence-interval half-width across independent training runs. Return AUC is the time-averaged return over 50k--500k steps.}",
             r"\label{tab:manip-runs}", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Task & Method & Return at 500k & Return AUC & Sampled ever-success (\%) \\",
             r"\midrule"]
    mapping = []
    for task_index, (task, title) in enumerate(TASKS):
        for method_index, (method, name) in enumerate(METHODS):
            group = sorted([r for r in records if r["panel"] == task and r["method"] == method],
                           key=lambda r: (r.get("replicate", r.get("seed")), r["run_id"]))
            if len(group) < 2:
                raise ValueError(f"At least two evaluations required for {task}/{method}")
            final_returns, aucs, successes = [], [], []
            for run_number, r in enumerate(group, 1):
                assert r["steps"] == list(range(50000, 500001, 50000))
                values = r["returns"]
                auc = sum((a + b) / 2 for a, b in zip(values, values[1:])) / 9
                assert abs(auc - r["metrics"]["return_auc"]) < 1e-6
                success = r["metrics"]["stoch_ever"]
                assert 0 <= success <= 100
                final_returns.append(values[-1])
                aucs.append(auc)
                successes.append(success)
                mapping.append({"task": task, "method": method, "display_run": run_number,
                                "replicate": r.get("replicate", r.get("seed")), "run_id": r["run_id"],
                                "evaluation_source": r["eval"], "return_500k": values[-1],
                                "return_auc": auc, "sampled_ever_success": success})
            row = [title if method_index == 0 else "", name,
                   table_cell(final_returns), table_cell(aucs), table_cell(successes)]
            lines.append(" & ".join(row) + r" \\")
        if task_index < len(TASKS) - 1:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, mapping = render(json.loads(args.input.read_text())["records"])
    (args.output_dir / "manipulation_runs.tex").write_text(table)
    report = {"sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "runs": mapping}
    (args.output_dir / "manipulation_runs.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
