"""Fixed-origin QSM evaluation for the narrow-navigation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import evaluate_multigoal_docking_full as base
import run_multigoal_docking_narrow_qsm as launcher

RUN_ID = re.compile(
    r"^docknav020_s(?P<sigma>003)_(?P<method>qsm)"
    r"_s(?P<seed>\d+)_n(?P<steps>\d+)(?P<smoke>_smoke)?$"
)


def evaluate(checkpoint, device="cpu", n=1000, eval_seed=1):
    contract = json.loads((Path(checkpoint).parent / "run_contract.json").read_text())
    if contract.get("geometry") != launcher.GEOMETRY:
        raise ValueError("checkpoint geometry differs from narrow campaign")
    original_run_id, original_launcher = base.RUN_ID, base.launcher
    try:
        base.RUN_ID, base.launcher = RUN_ID, launcher
        result = base.evaluate(checkpoint, device=device, n=n, eval_seed=eval_seed)
    finally:
        base.RUN_ID, base.launcher = original_run_id, original_launcher
    rows = result["episodes"]
    arrivals = [sum(row["goal"] == i for row in rows) for i in range(4)]
    hits = [sum(row["goal"] == i and row["success"] for row in rows) for i in range(4)]
    result.update(schema="multigoal_docking_narrow_eval_v1", geometry=launcher.GEOMETRY)
    result["metrics"].update(
        arrivals_by_goal=arrivals, docking_hits_by_goal=hits,
        arrival_rate=sum(arrivals) / n,
        conditional_docking_hit_by_goal=[h / a if a else None for h, a in zip(hits, arrivals)],
        timeout_rate=sum(row["goal"] < 0 for row in rows) / n,
        mean_navigation_steps=statistics.mean(row["navigation_steps"] for row in rows),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    cli = parser.parse_args()
    result = evaluate(cli.checkpoint, device=cli.device)
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with cli.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
