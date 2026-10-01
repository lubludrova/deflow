"""Frozen-policy manipulation evaluation with native success and dense return."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "repro"))
sys.path.insert(0, str(SRC / "analysis"))

from cluster_ci import summarize_rate
import run_manipulation_baseline as campaign
import run_manipulation_stage1 as stage1


ACTOR_MODULES = {
    "gated": SRC / "sac_anderson_flow.py",

    "gaussian": SRC / "sac_gaussian_baseline.py",
    "parent": SRC / "sac_flow_parent.py",
    "dime": SRC / "dime_actor.py",
}
ACTOR_DEFAULTS = {
    "gated": {"hidden_dim": 256, "max_iter_fwd": 25, "tol_fwd": 1e-5,
              "exact_backward": True, "denoising_steps": 4},
    "gaussian": {},
    "parent": {"denoising_steps": 4},
    "dime": {"denoising_steps": 16},
}


COMPARATOR_METHODS = {"gaussian": "sac", "parent": "sacflow", "dime": "dime"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(actor: str, obs_dim: int, act_dim: int, act_low, act_high,
                constructor: dict, device: torch.device, train_args: dict):
    constructor = {**ACTOR_DEFAULTS[actor], **constructor}
    if actor in COMPARATOR_METHODS:


        campaign.configure_method(COMPARATOR_METHODS[actor])
        actor_class = campaign.base.deq_multistep_flow_actor
        return actor_class(obs_dim, act_dim, torch.as_tensor(act_low),
                           torch.as_tensor(act_high), **constructor).to(device)
    wrapper = stage1.ACTOR_WRAPPERS[actor]({})
    if actor == "gated":
        engine_keys = ("denoising_steps", "integration", "density_estimator",
                       "gate_bias", "u_scale", "jac_sigma_target")
        for key in engine_keys:
            if key in train_args and key not in constructor:
                constructor[key] = train_args[key]
    return wrapper(obs_dim, act_dim, torch.as_tensor(act_low), torch.as_tensor(act_high),
                   **constructor).to(device)


def make_eval_env(env_alias: str, seed: int):
    if env_alias in stage1.STAGE3_ENV_RULES:
        spec = stage1.STAGE3_SPECS[env_alias]
        rules = stage1.STAGE3_ENV_RULES[env_alias]
    else:
        spec = campaign.ENVIRONMENTS[env_alias]
        rules = stage1.ENV_RULES[env_alias]
    __import__(spec["module"])
    if spec.get("construction_seed"):
        env = gym.make(spec["env_id"], construction_seed=seed)
    else:
        env = gym.make(spec["env_id"])
    env = stage1.Stage1Reward(env, rules["success_bonus"], 1.0)
    return env


def run_episode(actor, base_module, env, env_seed, policy_seed, channel,
                horizon, device, low, high):
    seed = env_seed if policy_seed is None else policy_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation, _ = env.reset(seed=env_seed)
    flags, rewards = [], []
    terminated = truncated = False
    for _ in range(horizon):
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1),
                                 device=device)
        with torch.no_grad():
            action = actor.act(tensor, deterministic=(channel == "det"))
        action = action.detach().cpu().numpy().astype(np.float32).reshape(-1)
        action = np.clip(action, low, high)
        observation, reward, terminated, truncated, info = env.step(action)
        native = base_module.native_success_vector(info, 1)
        if native is None:
            raise RuntimeError("environment exposed no native success signal")
        flags.append(bool(native[0]))
        rewards.append(float(reward))
        if terminated:
            break
    successful_steps = sum(flags)
    first = next((index + 1 for index, value in enumerate(flags) if value), None)
    return {
        "env_seed": env_seed, "policy_seed": policy_seed, "channel": channel,
        "steps": len(flags), "native_ever_success": any(flags),
        "success_at_horizon": flags[-1] if flags else False,
        "successful_steps": successful_steps,
        "first_success_step": first,
        "native_dense_return": sum(rewards),


        "terminated": bool(terminated), "truncated": bool(truncated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=stage1.RUN_ENVIRONMENTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--actor", choices=tuple(ACTOR_DEFAULTS), default=None)
    parser.add_argument("--mode", choices=("random", "panel"), default="random")
    parser.add_argument("--channels", default="stoch",
                        help="comma list of stoch and/or det")
    parser.add_argument("--environments", type=int, default=20)
    parser.add_argument("--samples-per-environment", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.set_defaults(seed=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.environments < 1 or args.samples_per_environment < 1:
        parser.error("evaluation counts must be positive")
    env_seeds = list(range(args.seed, args.seed + args.environments))
    policy_seeds = list(range(args.seed, args.seed + args.samples_per_environment))
    if args.mode == "panel":
        horizon = args.horizon or 50
        channels = ["det"]
    else:
        rule_table = (stage1.STAGE3_ENV_RULES if args.env in stage1.STAGE3_ENV_RULES
                      else stage1.ENV_RULES)
        horizon = args.horizon or rule_table[args.env]["horizon"]
        channels = [c for c in args.channels.split(",") if c]
    if any(c not in ("stoch", "det") for c in channels):
        parser.error(f"unsupported channels {channels}")

    contract = json.loads(args.contract.read_text()) if args.contract else {}
    actor_name = args.actor or contract.get("actor") or "gated"
    constructor = contract.get("actor_constructor") or {}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})

    env = make_eval_env(args.env, args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    device = torch.device(args.device)
    actor = build_actor(actor_name, obs_dim, act_dim, low, high, constructor,
                        device, train_args)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    base_module = sys.modules["sac_anderson_flow"]

    episodes = []
    for channel in channels:
        channel_policy_seeds = policy_seeds if channel == "stoch" else policy_seeds[:1]
        for env_seed in env_seeds:
            for policy_seed in channel_policy_seeds:
                episode_env = make_eval_env(args.env, env_seed)
                try:
                    episodes.append(run_episode(
                        actor, base_module, episode_env, env_seed, policy_seed,
                        channel, horizon, device, low, high))
                finally:
                    episode_env.close()

    aggregates = {}
    for channel in channels:
        channel_episodes = [ep for ep in episodes if ep["channel"] == channel]
        aggregates[channel] = {
            "ever_success": summarize_rate(channel_episodes, "native_ever_success",
                                           args.bootstrap, args.seed),
            "at_horizon": summarize_rate(channel_episodes, "success_at_horizon",
                                         args.bootstrap, args.seed),
            "mean_native_dense_return": statistics.fmean(
                ep["native_dense_return"] for ep in channel_episodes),
            "episodes": len(channel_episodes),
        }
    result = {
        "schema": "stage1_eval_v1",

        "mode": args.mode, "environment": args.env, "actor": actor_name,
        "channels": channels, "horizon": horizon,
        "env_seeds": env_seeds, "policy_seeds": policy_seeds,
        "protocol": ("Stochastic and deterministic evaluation"
                     if args.mode == "random" else "Deterministic evaluation panel"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "contract": str(args.contract.resolve()) if args.contract else None,
        "contract_sha256": sha256_file(args.contract) if args.contract else None,
        "actor_constructor": constructor,
        "actor_module_sha256": sha256_file(ACTOR_MODULES[actor_name]),
        "engine_module_sha256": sha256_file(SRC / "sac_anderson_flow.py"),
        "evaluator_module_sha256": sha256_file(Path(__file__)),
        "aggregate": aggregates,
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {channel: {
        "ever": aggregates[channel]["ever_success"]["rate"],
        "at_horizon": aggregates[channel]["at_horizon"]["rate"],
        "episodes": aggregates[channel]["episodes"],
        "ci_low": aggregates[channel]["ever_success"]["cluster_bootstrap"]["low"],
        "ci_high": aggregates[channel]["ever_success"]["cluster_bootstrap"]["high"],
    } for channel in channels}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
