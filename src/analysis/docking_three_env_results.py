"""Compute docking metrics from supplied checkpoint evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from published_results import docking_records

TASKS = ("Wide docking", "Tight docking", "Narrow navigation")
METHODS = ("sac", "sacflow", "dime", "qsm", "explicit", "deflow")


def load_rows(results_root):
    results_root = Path(results_root).resolve()
    rows = docking_records(results_root)
    if not rows:
        raise ValueError("No docking records supplied")
    identities = set()
    for row in rows:
        if row["method"] not in METHODS or row["status"] not in ("evaluated", "FAILED"):
            raise ValueError("Unsupported method or evaluation status")
        identity = (row["task"], row["method"], row.get("replicate", row.get("seed")))
        if identity in identities:
            raise ValueError(f"Duplicate evaluation identity: {identity}")
        identities.add(identity)
        row["data_root"] = str(results_root)
        curve = []
        for check in row["checkpoint_checks"]:
            path = Path(row["data_root"]) / check["evaluation"]
            raw = path.read_bytes()
            ev = json.loads(raw)
            assert ev["method"] == row["method"]
            assert ev.get("replicate", ev.get("seed")) == identity[2]
            assert int(path.stem.removeprefix("step_")) == ev["checkpoint_global_step"]
            assert ev["evaluation_start"] == [0., 0.]
            episodes = ev["episodes"]
            if not episodes:
                raise ValueError(f"No episodes: {path}")
            p = [sum(e["goal"] == g and e["success"] for e in episodes) / len(episodes) for g in range(4)]
            metrics = {"p_i": p, "u4": statistics.mean(1 - (1 - v) ** 4 for v in p),
                       "total_success": sum(p), "mean_return": statistics.mean(e["return"] for e in episodes)}
            for key in ("u4", "total_success", "mean_return"):
                assert abs(metrics[key] - ev["metrics"][key]) < 1e-10
            check["evaluation_sha256"] = hashlib.sha256(raw).hexdigest()
            curve.append({"step": ev["checkpoint_global_step"], **metrics})
        curve.sort(key=lambda item: item["step"])
        if len({point["step"] for point in curve}) != len(curve):
            raise ValueError("Repeated checkpoint step")
        row["curve"] = curve
        if row["status"] == "evaluated":
            if len(curve) < 2:
                raise ValueError("A completed curve needs at least two checkpoints")
            row.update({k: v for k, v in curve[-1].items() if k != "step"})
            row["auc_from_first_checkpoint"] = {"normalized": {"u4": sum(
                (b["step"] - a["step"]) * (a["u4"] + b["u4"]) / 2
                for a, b in zip(curve, curve[1:])) / (curve[-1]["step"] - curve[0]["step"])}}
    return rows


def cells(rows, tasks=TASKS):
    result = []
    for task in tasks:
        for method in METHODS:
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            finals = [r for r in group if r["status"] == "evaluated"]
            failed = [r for r in group if r["status"] == "FAILED"]
            mean = lambda f: statistics.mean(f(r) for r in finals) if finals else None
            result.append(dict(task=task, method=method, evaluated=len(finals), failed=len(failed),
                               completed_replicates=sorted(r.get("replicate", r.get("seed")) for r in finals),
                               failed_replicates=sorted(r.get("replicate", r.get("seed")) for r in failed),
                               coverage=mean(lambda r: r["u4"]),
                               sd=statistics.stdev(r["u4"] for r in finals) if len(finals) > 1 else None,
                               success=mean(lambda r: r["total_success"]),
                               mean_return=mean(lambda r: r["mean_return"]),
                               worst_goal=mean(lambda r: min(r["p_i"])),
                               balance_gap=mean(lambda r: 1 - (1 - r["total_success"] / 4) ** 4 - r["u4"]),
                               auc_u4=mean(lambda r: r["auc_from_first_checkpoint"]["normalized"]["u4"])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.results_root)
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "cells": cells(rows, tasks)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
