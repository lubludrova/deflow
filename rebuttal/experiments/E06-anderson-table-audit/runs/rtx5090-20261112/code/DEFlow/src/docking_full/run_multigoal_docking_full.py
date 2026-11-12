"""Train docking policies with real-transition step accounting."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_ROOT))

import multigoal_docking_env
import sac_anderson_flow as engine

DEFlowActor = engine.deq_multistep_flow_actor
ENVIRONMENTS = {"0.06": "MultiGoalDock-v0", "0.03": "MultiGoalDockTight-v0"}
METHODS = ("sac", "sacflow", "dime", "qsm", "explicit", "deflow")
SOURCE_FILES = (
    "sac_anderson_flow.py", "sac_flow_parent.py", "sac_gaussian_baseline.py",
    "dime_actor.py", "qsm_actor.py", "multigoal_docking_env.py",
    "step_accounting.py", "run_multigoal_docking_full.py",
    "evaluate_multigoal_docking_full.py",
)


def source_hashes():
    return {name: hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


class GatedDockingActor(DEFlowActor):


    def __init__(self, *args, **kwargs):
        kwargs.update(hidden_dim=256, max_iter_fwd=25, tol_fwd=1e-5,
                      exact_backward=True)
        super().__init__(*args, **kwargs)


def configure(method: str, sigma: str, seed: int, steps: int, smoke: bool = False,
              run_id: str | None = None):
    if method not in METHODS or sigma not in ENVIRONMENTS:
        raise ValueError("unknown method or docking sigma")
    if seed < 0 or steps <= 0 or steps % 8:
        raise ValueError("seed must be nonnegative and steps a positive multiple of 8")


    import sac_flow_parent as parent
    from sac_gaussian_baseline import gaussian_actor
    from dime_actor import PRIOR_STD, dime_dis_actor

    actor = {
        "sac": gaussian_actor,
        "sacflow": parent.parent_flow_actor,
        "dime": dime_dis_actor,
        "explicit": GatedDockingActor,
        "deflow": GatedDockingActor,
    }.get(method)
    if method != "qsm":
        engine.deq_multistep_flow_actor = actor
        engine.critic = parent.parent_critic

    if method == "qsm":
        from qsm_actor import QSMArgs
        args = QSMArgs()
        args.campaign_artifacts = True
        args.denoising_steps = 5
        args.M_q = 50.0

    else:
        args = engine.Args()
    run_id = run_id or (f"dockfull_s{sigma.replace('.', '')}_{method}_s{seed}_n{steps}"
                        + ("_smoke" if smoke else ""))
    args.run_name = args.exp_name = run_id
    args.env_id, args.seed = ENVIRONMENTS[sigma], seed
    args.num_envs = 8
    args.total_timesteps = steps
    args.learning_starts = 50_000
    args.batch_size = 512
    args.buffer_size = 1_000_000
    args.utd_ratio = 1.0
    args.policy_frequency = 1
    args.actor_lr, args.critic_lr, args.alpha_lr = 3e-4, 1e-3, 1e-3
    args.gamma, args.tau = 0.8, 0.005
    args.eval_interval = 1_024 if smoke else 10_000
    args.eval_envs = 5
    args.ckpt_interval = steps if smoke else 25_000
    args.save_model = True
    args.save_full_state = False
    args.resume_full_state = args.resume_ckpt = ""
    args.allow_cpu = smoke
    args.log_interval, args.tb_interval = 1_000, 1_000

    if method == "sacflow":
        args.alpha_init = 0.2
        args.target_entropy_scale = 0.0
    elif method == "dime":
        args.alpha_init = 1.0
        args.target_entropy_scale = 4.0 - (
            0.5 * math.log(2 * math.pi * math.e) + math.log(PRIOR_STD))
        args.denoising_steps = 16
        args.grad_clip = 0.0
    elif method in ("explicit", "deflow"):


        args.learning_starts = 4_000
        args.batch_size = 1_024
        args.buffer_size = 500_000
        args.utd_ratio = 0.5
        args.critic_lr = args.alpha_lr = 3e-4
        args.tau = 0.01
        args.alpha_init = 0.2
        args.target_entropy_scale = 0.0
        args.denoising_steps = 4
        args.integration = "explicit" if method == "explicit" else "implicit"
        args.density_estimator = "exact"
        args.u_scale, args.gate_bias = 4.0, 1.0
        args.grad_clip = 1.0
        args.lam_jac, args.jac_sigma_target = 0.01, 0.9
        args.jac_warmup = 100_000
    if smoke:
        args.learning_starts = 1_024
        args.batch_size = 64
        args.buffer_size = 20_000
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.set_defaults(seed=1)
    parser.add_argument("--sigma", choices=tuple(ENVIRONMENTS), default="0.06")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan", action="store_true", help="print recipe without training")
    cli = parser.parse_args()
    if cli.smoke and cli.steps is not None:
        parser.error("--smoke fixes the budget at 2048 interactions")
    steps = 2_048 if cli.smoke else (cli.steps or 200_000)
    try:
        args = configure(cli.method, cli.sigma, cli.seed, steps, cli.smoke)
    except ValueError as exc:
        parser.error(str(exc))

    if cli.method == "qsm":
        actor_name, critic_name = "qsm_actor.qsm_actor", "sac_flow_parent.parent_critic"
    else:
        actor_name = (f"{engine.deq_multistep_flow_actor.__module__}."
                      f"{engine.deq_multistep_flow_actor.__name__}")
        critic_name = f"{engine.critic.__module__}.{engine.critic.__name__}"
        expected = {
            "sac": "sac_gaussian_baseline.gaussian_actor",
            "sacflow": "sac_flow_parent.parent_flow_actor",
            "dime": "dime_actor.dime_dis_actor",
            "explicit": f"{__name__}.GatedDockingActor",
            "deflow": f"{__name__}.GatedDockingActor",
        }[cli.method]
        if actor_name != expected or critic_name != "sac_flow_parent.parent_critic":
            raise RuntimeError(f"actor/critic substitution: {actor_name}, {critic_name}")
    recipe = ("tent0" if cli.method == "deflow" else
              "tent0_explicit_euler" if cli.method == "explicit" else
              "method_canonical_matched_interactions")
    contract = {"run_id": args.run_name, "method": cli.method, "recipe": recipe,
                "docking_sigma": float(cli.sigma), "actor": actor_name,
                "critic": critic_name, "configuration": asdict(args),
                "runtime": "fresh NEXT_STEP; global_step counts replay transitions",
                "source_sha256": source_hashes()}
    if cli.plan:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return

    run_dir = Path("runs") / args.run_name
    run_dir.parent.mkdir(exist_ok=True)
    run_dir.mkdir()
    with (run_dir / "run_contract.json").open("x", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"identity-probe: actor={actor_name} critic={critic_name} run_id={args.run_name}")
    if cli.method == "qsm":
        import qsm_actor
        qsm_actor.main(args)
    else:
        engine.main(args)


if __name__ == "__main__":
    main()
