"""Validate published result identities and metrics without private run artifacts."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from continuous_control_preview import cohort
from docking_three_env_results import load_rows
from fig_manipulation_native import load
from published_results import solver_records


def read(path):
    return json.loads(path.read_text())


def check_results(root, tensorboard_python):
    docking = load_rows(root / "docking")
    manipulation, _ = load(root / "manipulation", tensorboard_python)
    mujoco = cohort(root / "mujoco")
    solver = solver_records(root / "numerical/solver_accuracy")
    for record in solver:
        base = root / "numerical/solver_accuracy"
        evaluations = {arm: read(base / path) for arm, path in record["evaluations"].items()}
        episode_keys = lambda ev: [(e["channel"], e["env_seed"], e["policy_seed"])
                                   for e in ev["episodes"]]
        assert episode_keys(evaluations["original"]) == episode_keys(evaluations["tight100"])
        benchmark = next(r for r in manipulation if r["panel"] == record["task"]
                         and r["method"] == "deflow" and r["seed"] == record["seed"])
        reference = read(root / "manipulation" / benchmark["eval"])
        for arm, ev in evaluations.items():
            assert ev["arm"] == arm and ev["seed"] == record["seed"]
            assert ev["checkpoint_global_step"] == reference["checkpoint_global_step"]
            assert ev["checkpoint_sha256"] == reference["checkpoint_sha256"]
            assert len(set(episode_keys(ev))) == len(ev["episodes"])
        audit = read(base / record["accuracy"])
        for row in audit["rows"]:
            assert row["arm"] in evaluations
            if row["reference_ok"]:
                assert all(np.isfinite(row[key]) and row[key] >= 0
                           for key in ("action_linf", "logp_abs_nats"))

    base = root / "numerical/integrator_depth"
    inputs = read(base / "shared_inputs.json")
    shared = read(base / "shared_results.json")["rows"]
    bank = inputs["banks"]["wide_tight"]
    shared_goals = np.repeat(bank["goals"], len(inputs["latents"]))
    targets = np.array([[.6, .2], [-.6, -.2], [-.2, .6], [.2, -.6]], dtype=np.float32)
    radius = np.float32(2 * bank["sigma"])
    panels = 0
    for method in ("explicit", "deflow"):
        own = read(base / f"{method}_own.json")
        assert own["method"] == method and own["replay_bit_identical"]
        donor = own["donor"]
        donor_run = root / "docking" / donor["run_id"]
        ev = read(donor_run / "evaluations" / f"step_{donor['checkpoint_global_step']:09d}.json")
        assert ev["method"] == method and ev["checkpoint_sha256"] == donor["checkpoint_sha256"]
        for steps in (2, 4, 8, 16):
            shared_row = next(r for r in shared if r["run_id"] == donor["run_id"]
                              and r["test_T"] == steps and r["geometry"] == "wide_tight")
            own_row = next(r for r in own["rows"] if r["test_T"] == steps)
            for row, goals, balanced in ((shared_row, shared_goals, True),
                                         (own_row, np.array(own_row["goals"]), False)):
                actions = np.asarray(row["actions"], dtype=np.float32)
                assert actions.shape == (len(goals), 2) and np.isfinite(actions).all()
                hits = np.linalg.norm(actions - targets[goals], axis=1) <= radius
                value = np.mean([hits[goals == g].mean() for g in range(4)]) if balanced else hits.mean()
                assert abs(value - row["balanced_hit" if balanced else "docking_hit"]) < 1e-7
                panels += 1
    return dict(docking_runs=len(docking),
                docking_evaluations=sum(len(row["curve"]) for row in docking),
                manipulation_runs=len(manipulation),
                mujoco_runs=sum(cell["n_runs"] for methods in mujoco.values() for cell in methods.values()),
                solver_runs=len(solver), integrator_panels=panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results")
    parser.add_argument("--tensorboard-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    counts = check_results(args.results_root, args.tensorboard_python)
    print("PASS: published result identities and metrics; " + json.dumps(counts))
    print("Checkpoint contents, training configurations, and original training seeds are not verified.")


if __name__ == "__main__":
    main()
