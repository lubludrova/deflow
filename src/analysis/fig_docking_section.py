"""Docking results from all valid completed runs, with failures shown separately."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from docking_three_env_results import cells as result_cells, load_rows

ROOT = Path(".")
OUT = Path(".")
METHODS = ("sac", "sacflow", "dime", "qsm", "explicit", "deflow")
NAMES = ("SAC", "SAC-Flow", "DIME", "QSM", "Explicit flow", "DEFlow")

COLORS = ("#7f7f7f", "#9467bd", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728")
TITLES = ("Wide docking", "Narrow + wide docking", "Tight docking", "Narrow navigation")
PANEL_TITLES = ("Wide navigation\nLoose docking", "Narrow navigation\nLoose docking",
                "Wide navigation\nTight docking", "Narrow navigation\nTight docking")
GOALS = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]])
TARGETS = np.array([[.6, .2], [-.6, -.2], [-.2, .6], [.2, -.6]])
GOAL_COLORS = ("#4c78a8", "#f2a541", "#59a14f", "#b279a2")
RAW_HASHES = {}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.22, lw=.5)
    ax.set_axisbelow(True)


def geometry():
    fig, axes = plt.subplots(2, 4, figsize=(10.8, 4.5))
    for col, title in enumerate(PANEL_TITLES):
        nav, dock = axes[:, col]
        for i, ((x, y), (a, b), color) in enumerate(zip(GOALS, TARGETS, GOAL_COLORS)):
            nav.add_patch(Ellipse((x, y), 2, .4 if col in (1, 3) else 2,
                                 angle=90 if x == 0 else 0,
                                 facecolor=color, edgecolor=color, alpha=.3))
            nav.annotate(str(i + 1), (x, y), (5, 5), textcoords="offset points", fontsize=9)
            dock.add_patch(Circle((a, b), .12 if col in (0, 1) else .06,
                                  facecolor=color, edgecolor=color, alpha=.3))
            dock.plot(a, b, "x", color=color, ms=5, mew=1.3)
            dock.annotate(str(i + 1), (a, b), (6, 5), textcoords="offset points", fontsize=9)
        nav.plot(0, 0, "*", color="#333333", ms=9)
        nav.set(xlim=(-6.5, 6.5), ylim=(-6.5, 6.5), aspect="equal")
        nav.set_xticks([-5, 0, 5]); nav.set_yticks([-5, 0, 5])
        nav.set_title(title, fontsize=11, pad=8)
        dock.set(xlim=(-.85, .85), ylim=(-.85, .85), aspect="equal")
        dock.set_xticks([-.6, 0, .6]); dock.set_yticks([-.6, 0, .6])
        dock.set_xlabel("Action $a_x$", fontsize=10)
        dock.text(.5, 1.03, "Docking radius: " + ("0.12" if col in (0, 1) else "0.06"),
                  transform=dock.transAxes, ha="center", fontsize=10)
        for ax in (nav, dock):
            style(ax)
    axes[0, 0].set_ylabel("Position $y$", fontsize=11)
    axes[1, 0].set_ylabel("Action $a_y$", fontsize=11)
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    save(fig, "fig_docking_task")


def raw_curve(row):
    curve = {}
    for check in row["checkpoint_checks"]:
        path = ROOT / row["data_root"] / check["evaluation"]
        raw = path.read_bytes()
        RAW_HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        episodes = data["episodes"]
        assert episodes
        p = np.array([sum(e["success"] and e["goal"] == g for e in episodes)
                      for g in range(4)]) / len(episodes)
        u4 = float(np.mean(1 - (1 - p) ** 4))
        assert abs(u4 - data["metrics"]["u4"]) < 1e-12
        step = data["checkpoint_global_step"]
        nominal = round(step / 25000) * 25000
        assert abs(step - nominal) < 100
        curve[nominal] = u4
    return curve


def learning(rows):
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.4))
    evidence = {}
    for ax, title in zip(axes, TITLES):
        ax.axhline(1 - .75 ** 4, color=".5", ls=":", lw=1)
        for method, color in zip(METHODS, COLORS):
            selected = [r for r in rows if r["task"] == title and r["method"] == method]
            completed = [r for r in selected if r["status"] == "evaluated"]
            curves = [raw_curve(r) for r in completed]
            points = []
            for step in range(25000, 200001, 25000):
                values = [c[step] for c in curves if step in c]
                if values:
                    assert len(values) == len(completed)
                    points.append((step, np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0., len(values)))
            failed = []
            for row in (r for r in selected if r["status"] == "FAILED"):
                curve = raw_curve(row)
                if not curve:
                    continue
                x, y = zip(*sorted(curve.items()))
                ax.plot(np.array(x) / 1000, y, color=color, lw=.9, ls="--", alpha=.7)
                ax.plot(x[-1] / 1000, y[-1], "x", color=color, ms=6)
                failed.append(dict(replicate=row.get("replicate", row.get("seed")), failure_transition=row.get("failure_transition"), curve=curve))
            evidence[f"{title}/{method}"] = dict(completed_replicates=[r.get("replicate", r.get("seed")) for r in completed],
                                                 points=points, failed=failed)
            if points:
                x, mean, sd, _ = np.array(points).T
                ax.plot(x / 1000, mean, color=color, lw=2)
                ax.fill_between(x / 1000, np.maximum(0, mean - sd),
                                np.minimum(1 - .75 ** 4, mean + sd), color=color, alpha=.15, lw=0)
        ax.set(xlim=(25, 203), ylim=(-.025, .72), xlabel="Transitions (k)")
        ax.set_title(PANEL_TITLES[TITLES.index(title)], fontsize=11)
        ax.set_xticks([25, 100, 200])
        style(ax)
    axes[0].set_ylabel("Successful coverage (U4)")
    handles = [plt.Line2D([], [], color=c, lw=2.2, label=n)
               for n, c in zip(NAMES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(.5, .055), fontsize=10)
    fig.tight_layout(rect=(0, .19, 1, 1))
    save(fig, "fig_docking_learning")
    return evidence


def table(rows):
    lines = [r"{\scriptsize\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Navigation & Docking & SAC & SAC-Flow & DIME & QSM & Explicit flow & DEFlow \\",
             r"\midrule"]
    evidence = result_cells(rows, tasks=TITLES)
    for title in TITLES:
        groups = [g for g in evidence if g["task"] == title]
        ranked = sorted({g["coverage"] for g in groups if g["coverage"] is not None}, reverse=True)
        cells = []
        for g in groups:
            coverage = "---" if g["coverage"] is None else f'{g["coverage"]:.3f}'
            if ranked and g["coverage"] == ranked[0]:
                coverage = r"\mathbf{" + coverage + "}"
            elif len(ranked) > 1 and g["coverage"] == ranked[1]:
                coverage = r"\underline{" + coverage + "}"
            cells.append("---" if g["coverage"] is None else "$" + coverage + r"\pm" + (f'{g["sd"]:.3f}' if g["sd"] is not None else "--") + "$")
        navigation, docking = PANEL_TITLES[TITLES.index(title)].split("\n")
        label = navigation.removesuffix(" navigation") + " & " + docking.removesuffix(" docking")
        lines.append(label + " & " + " & ".join(cells) + " " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", "}"]
    (OUT / "docking_results.tex").write_text("\n".join(lines) + "\n")
    return evidence


def main():
    global ROOT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ROOT, OUT = args.results_root.resolve(), args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.linewidth": .8, "pdf.fonttype": 42})
    rows = load_rows(args.results_root)
    geometry()
    curves = learning(rows)
    summaries = table(rows)
    evidence = dict(status_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in sorted(ROOT.glob("*/*/seed_*/status.json"))},
                    raw_evaluation_sha256=RAW_HASHES, cells=summaries, curves=curves)
    (OUT / "docking_data.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
