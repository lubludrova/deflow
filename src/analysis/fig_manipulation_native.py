"""Generate manipulation exhibits for the paper cohort, including native-continuing Peg."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from published_results import manipulation_records

METHODS = ["sac", "sacflow", "dime", "deflow"]
NAMES = ["SAC", "SAC-Flow", "DIME", "DEFlow (ours)"]
COLORS = ["#7f7f7f", "#9467bd", "#ff7f0e", "#d62728"]
PANELS = ["ms_pickcube", "ms_pushcube", "mw_button", "mw_peg_native", "mw_pushwall"]
TITLES = ["MS PickCube", "MS PushCube", "MW ButtonPressWall",
          "MW PegInsertSide", "MW PushWall"]
PROTOCOLS = ["H=50; continuing", "H=50; continuing", "H=200; success-stop",
             "H=200; native-continuing", "H=200; success-stop"]
GRID = np.arange(50000, 500001, 50000)
METRICS = ["stoch_ever", "return_auc", "success_auc", "stoch_end"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def auc(values):
    values = np.asarray(values)
    assert values.shape == (10,) and np.isfinite(values).all()
    return float(np.sum(np.diff(GRID) * (values[:-1] + values[1:]) / 2) / 450000)


def group(records, panel, method):
    return [r for r in records if r["panel"] == panel and r["method"] == method]


def stats(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "median": float(np.median(values)), "values": list(values), "n": len(values)}


def load(results_root, tb_python):
    source = Path(results_root).resolve()
    records = manipulation_records(source)
    if not records:
        raise ValueError("No evaluation records supplied")
    if any(r["panel"] not in PANELS or r["method"] not in METHODS for r in records):
        raise ValueError("Unsupported task or method")
    paths = {}
    reader = '''import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
output=[]
for paths in json.load(sys.stdin):
    tags={"returns":{}, "success":{}}
    for path in paths:
        acc=EventAccumulator(path, size_guidance={"scalars":0}); acc.Reload()
        for key, tag in [("returns","charts/eval_return_det"),("success","charts/eval_native_success_rate_det")]:
            for event in acc.Scalars(tag):
                if event.step in range(50000,500001,50000):
                    value=event.value*(100 if key=="success" else 1)
                    if event.step in tags[key]:
                        assert tags[key][event.step]==value
                    tags[key][event.step]=value
    assert all(sorted(points)==list(range(50000,500001,50000)) for points in tags.values())
    output.append({key:[points[s] for s in sorted(points)] for key,points in tags.items()})
print(json.dumps(output))
'''
    events = [[str(source / p) for p in r["events"]] for r in records]
    raw = subprocess.run([str(tb_python), "-c", reader], input=json.dumps(events),
                         text=True, capture_output=True, check=True)
    for record, curves in zip(records, json.loads(raw.stdout)):
        for key in ("returns", "success"):
            if key in record:
                np.testing.assert_allclose(record[key], curves[key], rtol=0, atol=1e-6)
            record[key] = curves[key]
        record["steps"] = GRID.tolist()
        ev = json.loads((source / record["eval"]).read_text())
        assert ev["checkpoint_global_step"] == 500000
        assert ev["method"] == record["method"] and ev["seed"] == record["seed"]
        expected_environment = "mw_peg" if record["panel"] == "mw_peg_native" else record["panel"]
        assert ev["environment"] == expected_environment
        metrics = {"return_auc": auc(record["returns"]), "success_auc": auc(record["success"])}
        for channel in ("stoch", "det"):
            eps = [e for e in ev["episodes"] if e["channel"] == channel]
            count = len(eps)
            if not count:
                raise ValueError(f"No {channel} episodes in {record['eval']}")
            assert len({(e["env_seed"], e["policy_seed"]) for e in eps}) == count
            for name, key, aggkey in [("ever", "native_ever_success", "ever_success"),
                                      ("end", "success_at_horizon", "at_horizon")]:
                value = 100 * float(np.mean([e[key] for e in eps]))
                assert np.isclose(value, 100 * ev["aggregate"][channel][aggkey]["rate"])
                metrics[channel + "_" + name] = value
            metrics[channel + "_return"] = float(np.mean([e["native_dense_return"] for e in eps]))
        if "metrics" in record:
            for key, value in metrics.items():
                np.testing.assert_allclose(value, record["metrics"][key], rtol=0, atol=1e-6)
        record["metrics"] = metrics
        for name in [record["eval"], *record["events"]]:
            paths[name] = digest(source / name)
    for panel in PANELS:
        for method in METHODS:
            cell = group(records, panel, method)
            if len(cell) < 2:
                raise ValueError(f"At least two evaluations required for {panel}/{method}")
            assert len({r.get("replicate", r.get("seed")) for r in cell}) == len(cell)
    return records, paths


def summary(records):
    result = {p: {m: {key: stats([r["metrics"][key] for r in group(records, p, m)])
                       for key in METRICS} for m in METHODS} for p in PANELS}
    leaders = {m: [] for m in METHODS}
    for p in PANELS:
        means = np.array([np.mean([r["returns"] for r in group(records, p, m)], axis=0) for m in METHODS])
        for i, m in enumerate(METHODS):
            leaders[m].append(int(np.sum(np.argmax(means, axis=0) == i)))
    return result, leaders


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.22, lw=.5)
    ax.set_axisbelow(True)


def save(fig, output, stem):
    for ext in ("png", "pdf"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(output / (stem + "." + ext), dpi=220, bbox_inches="tight", **kwargs)
    plt.close(fig)


def legend(fig, y=.03, fontsize=11):
    fig.legend(handles=[plt.Line2D([], [], color=c, lw=2.2, label=n) for c, n in zip(COLORS, NAMES)],
               loc="lower center", bbox_to_anchor=(.5, y), ncol=4, frameon=False, fontsize=fontsize)


def ranks(values):
    distinct = sorted(set(round(v, 8) for v in values), reverse=True)
    if not distinct or distinct[0] == 0:
        return [0] * len(values)
    return [distinct.index(round(v, 8)) + 1 for v in values]


def return_figure(records, output):
    fig, axes = plt.subplots(1, 5, figsize=(12.8, 3.5))
    for ax, panel, title in zip(axes, PANELS, TITLES):
        style(ax)
        for method, color in zip(METHODS, COLORS):
            values = np.asarray([r["returns"] for r in group(records, panel, method)])
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(GRID / 1e6, mean, color=color, lw=2)
            ax.plot([0, .05], [0, mean[0]], color=color, lw=1, ls="--")
            ax.fill_between(GRID / 1e6, mean - sd, mean + sd, color=color, alpha=.15, lw=0)
        ax.set_title(title.replace(" ", "\n", 1))
        ax.set(xlim=(0, .5), xticks=[0, .25, .5], xlabel="Steps (M)")
    axes[0].set_ylabel("Evaluation return")
    legend(fig, .01)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, output, "fig_manipulation_native_return")


def paper_table(cells, metric):
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Task & SAC & SAC-Flow & DIME & DEFlow \\", r"\midrule"]
    totals = [sum(cells[p][m][metric]["mean"] for p in PANELS) for m in METHODS]
    for panel, title in zip([*PANELS, None], [*TITLES, "Sum"]):
        values = totals if panel is None else [cells[panel][m][metric]["mean"] for m in METHODS]
        places = ranks(values)
        row = [title.replace("MS ", "").replace("MW ", "")]
        for method, value, place in zip(METHODS, values, places):
            text = f"{value:.1f}"
            if panel is not None:
                text += r" \pm " + f"{cells[panel][method][metric]['sd']:.1f}"
            if place == 1:
                text = r"\mathbf{" + text + "}"
            elif place == 2:
                text = r"\underline{" + text + "}"
            row.append("$" + text + "$")
        rows.append(" & ".join(row) + r" \\")
    return "\n".join(rows + [r"\bottomrule", r"\end{tabular}", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tensorboard-python", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, inputs = load(args.results_root, args.tensorboard_python)
    cells, leaders = summary(records)
    dump(args.output_dir / "data.json", {"records": records, "summary": cells,
        "leading_checkpoints": leaders, "source_files": inputs})
    paper = args.output_dir
    figures = paper / "Figures"
    figures.mkdir(parents=True, exist_ok=True)
    return_figure(records, figures)
    for name, metric in [("manipulation_sampled_success_wide", "stoch_ever"),
                         ("manipulation_native_return_auc", "return_auc")]:
        (paper / (name + ".tex")).write_text(paper_table(cells, metric))
    print(f"Processed {len(records)} evaluations.")


if __name__ == "__main__":
    main()
