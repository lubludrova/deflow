"""Compare return summaries from the published result files."""

import argparse
import json
from pathlib import Path

from continuous_control_preview import selected_runs
from paper_return_report import sensitivity_table, summary
from return_metrics import ENVS, METRICS


def build_cells(results_root):
    cells = selected_runs(results_root)
    for env in cells:
        for model, runs in cells[env].items():
            for index, run in enumerate(runs, 1):
                run["display_id"] = f"R{index}"
            cells[env][model] = {"runs": runs, "summary": summary(runs)}
    return cells


def deflow_table(cells):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Environment & Run & Return at 1M & Last 100k mean & Return AUC & Return at 950k \\",
             r"\midrule"]
    for env, title in ENVS:
        if env not in cells or "flowe" not in cells[env]:
            continue
        for index, run in enumerate(cells[env]["flowe"]["runs"]):
            values = [run["scores"][m] for m in METRICS] + [run["return_950k"]]
            row = [title if index == 0 else "", run["display_id"]]
            row += ["--" if value is None else f"{value:.0f}" for value in values]
            lines.append(" & ".join(row) + r" \\")
        if env != ENVS[-1][0]:
            lines.append(r"\midrule")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cells = build_cells(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "continuous_control_sensitivity.tex").write_text(sensitivity_table(cells))
    (args.output_dir / "deflow_run_sensitivity.tex").write_text(deflow_table(cells))
    report = {"missing": "No interpolation; each metric requires its complete evaluation grid", "cells": cells}
    (args.output_dir / "continuous_control_sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
