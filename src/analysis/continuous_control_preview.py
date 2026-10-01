"""Plot continuous-control returns from the published result files."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from return_metrics import ENVS, GRID, MODELS, load_runs, observed
from published_results import mujoco_records

def selected_runs(results_root):
    results_root = Path(results_root)
    records = mujoco_records(results_root)
    cells = {}
    for record in records:
        env, method = record["environment"], record["method"]
        if env not in dict(ENVS) or method not in {row[0] for row in MODELS}:
            raise ValueError(f"Unsupported environment or method: {env}/{method}")
        group = cells.setdefault(env, {}).setdefault(method, [])
        identity = record.get("replicate", record.get("seed"))
        if identity is None:
            raise ValueError("Missing replicate identity")
        if any(run.get("replicate", run.get("seed")) == identity for run in group):
            raise ValueError(f"Duplicate replicate in {env}/{method}")
        path = results_root / record["path"]
        run = load_runs(results_root, [record])[0]
        if "replicate" in record:
            run.pop("seed", None)
            run["replicate"] = identity
        if run["scores"]["final"] is None:
            raise ValueError(f"Selected evaluation has no final checkpoint: {path}")
        group.append(run)
    if not cells:
        raise ValueError("No evaluation records supplied")
    return cells


def cohort(results_root):
    cells = selected_runs(results_root)
    for env in cells:
        for key, runs in cells[env].items():
            values = np.array([observed(r["steps"], r["returns"], GRID) for r in runs])
            mask = np.isfinite(values).all(axis=0)
            final = values[:, -1]
            cells[env][key] = {"runs": runs, "n_runs": len(runs),
                               "steps": GRID[mask].tolist(),
                               "mean": values[:, mask].mean(axis=0).tolist(),
                               "sd": (values[:, mask].std(axis=0, ddof=1).tolist() if len(runs) > 1 else None),
                               "final_values": final.tolist(), "final_mean": float(final.mean()),
                               "final_sd": float(final.std(ddof=1)) if len(runs) > 1 else None}
    return cells


def figure(cells, output):
    plt.rcParams.update({"font.size": 10, "axes.linewidth": .8})
    environments = [(key, title) for key, title in ENVS if key in cells]
    fig, axes = plt.subplots(1, len(environments), figsize=(3 * len(environments), 4.1), squeeze=False)
    axes = axes[0]
    for ax, (env, title) in zip(axes, environments):
        for key, _, color in MODELS:
            if key not in cells[env]:
                continue
            cell = cells[env][key]
            x = np.array(cell["steps"]) / 1e6
            mean = np.array(cell["mean"])
            sd = np.zeros_like(mean) if cell["sd"] is None else np.array(cell["sd"])
            ax.plot(x, mean, color=color, lw=2, zorder=4)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.15, lw=0)
        ax.set_title(title + "-v4", fontsize=15, pad=10)
        ax.set_xlabel("Steps (M)", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.tick_params(labelsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.22, lw=.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Evaluation return", fontsize=14)
    handles = [plt.Line2D([], [], color=c, lw=2.2,
                         label=n)
               for k, n, c in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=5, frameon=False, fontsize=14)
    fig.tight_layout(rect=(0, .15, 1, 1), w_pad=1)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def table(cells):
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            "Environment & " + " & ".join(name for _, name, _ in MODELS) + r" \\", r"\midrule"]
    for env, title in ENVS:
        if env not in cells:
            continue
        values = []
        for key, _, _ in MODELS:
            cell = cells[env].get(key)
            if cell is None:
                values.append("--")
                continue
            text = f"{cell['final_mean']:.0f}"
            if cell["final_sd"] is not None:
                text += r" \pm " + f"{cell['final_sd']:.0f}"
            values.append("$" + text + "$")
        rows.append(" & ".join([title, *values]) + r" \\")
    return "\n".join(rows + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = cohort(args.results_root)
    figure(cells, args.output_dir / "continuous_control")
    (args.output_dir / "continuous_control.tex").write_text(table(cells))
    (args.output_dir / "continuous_control.json").write_text(json.dumps({
        "cells": cells, "metric": "Recorded deterministic return at the final evaluation",
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
