"""Probe a frozen QSM policy at the four ideal docking observations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1] / "docking_full"
sys.path.insert(0, str(SOURCE))
from multigoal_docking_env import MultiGoalDockingEnv
from qsm_actor import qsm_actor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.set_defaults(seed=1)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    checkpoint_sha = digest(args.checkpoint)
    if checkpoint_sha != reference["checkpoint_sha256"] or reference["method"] != "qsm":
        raise ValueError("checkpoint or method differs from reference evaluation")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (checkpoint["actor_class"] != "qsm_actor.qsm_actor"
            or checkpoint["global_step"] != reference["checkpoint_global_step"]):
        raise ValueError("QSM checkpoint identity differs from reference evaluation")

    env = MultiGoalDockingEnv(docking_sigma=reference["sigma"])
    actor = qsm_actor(5, 2, [-1, -1], [1, 1], **checkpoint["actor_constructor"])
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    torch.set_num_threads(2)
    goal_results = []
    for goal, (position, target) in enumerate(zip(env.goals, env.docking_targets)):
        observation = np.r_[position, 1.0, position].astype(np.float32)
        torch.manual_seed(args.seed + goal)
        actions = actor.sample(torch.as_tensor(observation).repeat(args.samples, 1)).detach().numpy()
        distances = np.linalg.norm(actions - target, axis=1)
        goal_results.append({
            "goal": goal,
            "observation": observation.tolist(),
            "target": target.tolist(),
            "sampler_seed": args.seed + goal,
            "actions": actions.tolist(),
            "both_coordinates_clipped_fraction": float(np.mean(np.all(np.abs(actions) > .999, axis=1))),
            "median_target_distance": float(np.median(distances)),
            "min_target_distance": float(np.min(distances)),
            "docking_hits": int(np.count_nonzero(distances <= 2 * env.docking_sigma)),
        })
    report = {
        "schema": "docking_qsm_ideal_probe_v1",
        "source_sha256": {name: digest(SOURCE / name)
                          for name in ("qsm_actor.py", "multigoal_docking_env.py")},
        "probe_sha256": digest(Path(__file__)),
        "checkpoint_sha256": checkpoint_sha,
        "reference_eval_sha256": digest(args.reference),
        "run_id": checkpoint["run_id"],
        "checkpoint_global_step": checkpoint["global_step"],
        "actor_constructor": checkpoint["actor_constructor"],
        "samples_per_goal": args.samples,
        "docking_radius": 2 * env.docking_sigma,
        "goal_results": goal_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")
    for row in goal_results:
        print(row["goal"], row["both_coordinates_clipped_fraction"],
              row["median_target_distance"], row["min_target_distance"], row["docking_hits"])


if __name__ == "__main__":
    main()
