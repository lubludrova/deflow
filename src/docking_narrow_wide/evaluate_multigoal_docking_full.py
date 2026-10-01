"""Reconstruct docking policies and evaluate stochastic goal coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import gymnasium as gym
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import multigoal_docking_env
import run_multigoal_docking_full as launcher


RUN_ID = re.compile(
    r"^dockfull_s(?P<sigma>006|003)_(?P<method>sac|sacflow|dime|qsm|explicit|deflow)"
    r"_s(?P<seed>\d+)_n(?P<steps>\d+)(?P<smoke>_smoke)?$"
)
EVAL_SEED = 1
EPISODES = 1000


def summarize(episodes):

    n = len(episodes)
    if not n:
        raise ValueError("at least one episode is required")
    hits = np.bincount(
        [row["goal"] for row in episodes if row["success"]], minlength=4
    )
    p = hits / n
    total = float(p.sum())
    conditional = hits / hits.sum() if hits.sum() else np.zeros(4)
    positive = conditional[conditional > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "p_i": p.tolist(),
        "total_success": total,
        "successful_goal_entropy_nats": entropy,
        "successful_goal_entropy_normalized": entropy / math.log(4),
        "u4": float(np.mean(1.0 - (1.0 - p) ** 4)),
        "u16": float(np.mean(1.0 - (1.0 - p) ** 16)),
        "mean_return": float(np.mean([row["return"] for row in episodes])),
    }


def load_actor(checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args")
    if not isinstance(saved, dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError("checkpoint requires args and actor state")
    match = RUN_ID.fullmatch(saved.get("exp_name", ""))
    if match is None or saved.get("run_name") != saved.get("exp_name"):
        raise ValueError("checkpoint is not from the docking launcher")
    method = match["method"]
    sigma = "0.06" if match["sigma"] == "006" else "0.03"
    seed, steps = int(match["seed"]), int(match["steps"])
    configured = launcher.configure(
        method, sigma, seed, steps, smoke=bool(match["smoke"]),
        run_id=saved["exp_name"],
    )
    expected = vars(configured)
    for key in ("env_id", "seed", "total_timesteps", "denoising_steps", "integration",
                "density_estimator", "u_scale", "gate_bias", "lam_jac", "jac_sigma_target"):
        if saved.get(key) != expected.get(key):
            raise ValueError(f"checkpoint {key} differs from launcher recipe")
    if method == "qsm" and checkpoint.get("actor_class", "qsm_actor.qsm_actor") != "qsm_actor.qsm_actor":
        raise ValueError("QSM checkpoint actor identity differs")

    env = gym.make(saved["env_id"])
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
    finally:
        env.close()
    if method == "qsm":
        from qsm_actor import qsm_actor

        constructor = checkpoint.get("actor_constructor", {
            "T": saved["denoising_steps"], "M_q": saved["M_q"],
        })
        actor = qsm_actor(obs_dim, act_dim, low, high, **constructor)
    else:
        actor = launcher.engine.deq_multistep_flow_actor(
            obs_dim, act_dim, low, high,
            denoising_steps=saved["denoising_steps"],
            integration=saved["integration"],
            density_estimator=saved["density_estimator"],
            density_diag_every=saved["density_diag_every"],
            density_diag_batch=saved["density_diag_batch"],


            gate_bias=saved["gate_bias"],
            jac_sigma_target=(saved["jac_sigma_target"] if saved["lam_jac"] > 0 else 0.0),
            hidden_dim=128, u_scale=saved["u_scale"],
            max_iter_fwd=15, tol_fwd=1e-3, lam_fwd=1e-2, beta_fwd=0.7,
            max_iter_bwd=10, tol_bwd=1e-3, lam_bwd=1e-2, beta_bwd=0.7,
        )
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.to(device).eval()
    return actor, method, sigma, seed, saved["env_id"], checkpoint.get("global_step"), saved


def rollout(actor, method, env_id, device, n=EPISODES, eval_seed=EVAL_SEED):

    envs = [gym.make(env_id) for _ in range(n)]
    try:
        obs = np.stack([
            env.reset(seed=eval_seed + i, options={"fixed_origin": True})[0]
            for i, env in enumerate(envs)
        ])
        active = np.ones(n, dtype=bool)
        returns = np.zeros(n, dtype=np.float64)
        rows = [None] * n
        with torch.no_grad():
            while active.any():
                indexes = np.flatnonzero(active)
                for start in range(0, len(indexes), 128):
                    batch = indexes[start:start + 128]
                    tensor = torch.as_tensor(obs[batch], device=device)
                    actions = actor.act(tensor, deterministic=(method == "qsm"))
                    actions = actions.detach().cpu().numpy()
                    if not np.isfinite(actions).all():
                        raise FloatingPointError("nonfinite evaluation action")
                    for index, action in zip(batch, actions):
                        next_obs, reward, terminated, truncated, info = envs[index].step(action)
                        returns[index] += reward
                        obs[index] = next_obs
                        if terminated or truncated:
                            active[index] = False
                            rows[index] = {
                                "episode": int(index),
                                "goal": int(info["selected_goal"]),
                                "success": int(info["success"]),
                                "return": float(returns[index]),
                                "docking_distance": info.get("docking_distance"),
                                "navigation_steps": envs[index].unwrapped.navigation_steps,
                            }
        return rows
    finally:
        for env in envs:
            env.close()


def evaluate(checkpoint_path, device="cpu", n=EPISODES, eval_seed=EVAL_SEED):
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    contract_path = checkpoint_path.parent / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    source_sha256 = launcher.source_hashes()
    if contract.get("source_sha256") != source_sha256:
        raise ValueError("full-campaign source differs from the run contract")
    with checkpoint_path.open("rb") as source:
        checkpoint_sha256 = hashlib.sha256(source.read()).hexdigest()
        source.seek(0)
        with torch.random.fork_rng(devices=([device.index or 0] if device.type == "cuda" else [])):
            actor, method, sigma, seed, env_id, global_step, saved = load_actor(source, device)
            if (contract.get("run_id") != saved["exp_name"]
                    or contract.get("configuration") != saved):
                raise ValueError("run contract differs from checkpoint args")
            torch.manual_seed(eval_seed)
            episodes = rollout(actor, method, env_id, device, n, eval_seed)
    return {
        "schema": "multigoal_docking_full_eval_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "run_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "source_sha256": source_sha256,
        "checkpoint_global_step": global_step,
        "method": method,
        "sigma": float(sigma),
        "seed": seed,
        "environment": env_id,
        "evaluation_seed": eval_seed,
        "evaluation_start": [0.0, 0.0],
        "policy": "stochastic; QSM ancestral sampler without behavior noise",
        "n_episodes": n,
        "metrics": summarize(episodes),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    cli = parser.parse_args()
    result = evaluate(cli.checkpoint, device=cli.device)
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    with cli.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("method", "sigma", "seed", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
