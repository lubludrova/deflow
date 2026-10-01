"""Train the selected manipulation policies and record their configurations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import gymnasium as gym
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_manipulation_baseline as campaign

base = campaign.base


ARCH = {
    "gamma": 0.8,
    "utd_ratio": 0.5,
    "batch_size": 1024,
    "learning_starts": 4_000,
    "tau": 0.01,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "ent_track": False,
    "alpha_leak": 0.0,
    "lp_winsor": 0.0,
    "alpha_min": 0.0,
    "alpha_init": 0.2,
    "target_entropy_scale": 1.0,
    "target_entropy_scale_final": 0.0,
    "logp_clip": 0.0,
    "grad_clip": 1.0,
    "lam_jac": 1e-2,
    "jac_sigma_target": 0.9,
    "jac_warmup": 100_000,
    "buffer_size": 500_000,
    "policy_frequency": 1,
}

RECIPES = {
    "tent0": {**ARCH, "target_entropy_scale": 0.0},
    "r1": {**ARCH, "gamma": 0.99, "target_entropy_scale": 0.0},
    "r2hw": {**ARCH, "gamma": 0.99, "target_entropy_scale": 0.0,
             "target_entropy_scale_final": 0.5, "anneal_steps": 150_000,
             "alpha_leak": 1e-2, "alpha_prior": 0.05, "lp_winsor": 0.05,
             "logp_clip": 15.0, "alpha_min": 0.02, "ent_track": True},
    "sac": {"gamma": 0.8, "batch_size": 512, "learning_starts": 50_000,
            "policy_frequency": 1, "utd_ratio": 1.0},
    "sacflow": {"gamma": 0.8, "batch_size": 512, "learning_starts": 50_000,
                "policy_frequency": 1, "utd_ratio": 1.0},
    "dime": {"gamma": 0.8, "batch_size": 512, "learning_starts": 50_000,
             "policy_frequency": 1, "utd_ratio": 1.0},
    "sacsoft": {"gamma": 0.99, "batch_size": 512, "learning_starts": 50_000,
                "policy_frequency": 1, "utd_ratio": 1.0},
    "sacflowsoft": {"gamma": 0.99, "batch_size": 512, "learning_starts": 50_000,
                    "policy_frequency": 1, "utd_ratio": 1.0},
    "dimesoft": {"gamma": 0.99, "batch_size": 512, "learning_starts": 50_000,
                 "policy_frequency": 1, "utd_ratio": 1.0},
}
RECIPE_ACTOR = {name: "gated" for name in ("tent0", "r1", "r2hw")}
RECIPE_ACTOR.update({"sac": "gaussian", "sacsoft": "gaussian",
                     "sacflow": "parent", "sacflowsoft": "parent",
                     "dime": "dime", "dimesoft": "dime"})
RECIPE_METHOD = {"sac": "sac", "sacflow": "sacflow", "dime": "dime",
                 "sacsoft": "sac", "sacflowsoft": "sacflow", "dimesoft": "dime"}
RECIPE_NORMALIZED = {name: True for name in RECIPES}
RECIPE_SUCCESS_REWARD = {}
RECIPE_PRECISION = {}
RECIPE_NO_TB = {}
RECIPE_ACTOR_KWARGS = {}

DETERMINISM_ENV_VARS = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "PYTHONMALLOC",
                        "MALLOC_CHECK_", "CUDA_LAUNCH_BLOCKING", "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                        "SACFLOW_MANISKILL_RENDER_BACKEND")


_EFFECTIVE_PRECISION: dict = {}


PRISTINE_GATED_ACTOR = base.deq_multistep_flow_actor


_EFFECTIVE_ACTOR: dict = {}


def _capture_effective_actor(actor) -> None:
    output = getattr(getattr(actor, "vf", None), "pre", None)
    _EFFECTIVE_ACTOR.clear()
    _EFFECTIVE_ACTOR.update({
        "tol_fwd": float(getattr(actor, "tol_fwd", float("nan"))),
        "denoising_steps": int(getattr(actor, "T", -1)),
        "hidden_dim": int(output.out_features) if output is not None else None,
        "max_iter_fwd": int(getattr(actor, "max_iter_fwd", -1)),
        "exact_backward": bool(getattr(actor, "exact_backward", False)),
        "integration": getattr(actor, "integration", None),
        "density_estimator": getattr(actor, "density_estimator", None),
        "actor_class": type(actor).__name__,
    })


ENV_RULES = {
    "ms_pickcube": {"divisor": 5.0, "success_bonus": 1.0, "horizon": 50},
    "ms_pushcube": {"divisor": 4.0, "success_bonus": 1.0, "horizon": 50},
    "mw_peg": {"divisor": 10.0, "success_bonus": 0.0, "horizon": 500},
}
STAGE1_ENVIRONMENTS = tuple(ENV_RULES)


STAGE3_SPECS = {
    "mw_peg": {"env_id": "MetaWorldSoftPegInsertSide-v0",
               "module": "metaworld_soft_env", "success_threshold": 100.0,
               "construction_seed": True},
    "mw_button": {"env_id": "MetaWorldSoftButtonPressWall-v0",
                  "module": "metaworld_soft_env", "success_threshold": 100.0,
                  "construction_seed": True},
    "mw_pushwall": {"env_id": "MetaWorldSoftPushWall-v0",
                    "module": "metaworld_soft_env", "success_threshold": 100.0,
                    "construction_seed": True},
    "ms_pushcube": {"env_id": "MS3PushCube-v0", "module": "maniskill_env",
                    "success_threshold": 0.5},
}
_MW_SOFT_RULES = {"divisor": 1.0, "success_bonus": 0.0, "horizon": 200,
                  "terminal_bonus": 0.0, "terminate_on_success": True}
STAGE3_ENV_RULES = {
    "mw_peg": dict(_MW_SOFT_RULES),
    "mw_button": dict(_MW_SOFT_RULES),
    "mw_pushwall": dict(_MW_SOFT_RULES),
    "ms_pushcube": {"divisor": 4.0, "success_bonus": 1.0, "horizon": 50,
                    "terminal_bonus": 1.0, "terminate_on_success": True},
}
RUN_ENVIRONMENTS = tuple(sorted(set(STAGE1_ENVIRONMENTS) | set(STAGE3_ENV_RULES)))

CHUNK = 25_000
SMOKE_STEPS = 2_048


class Stage1Reward(gym.Wrapper):


    def __init__(self, env, success_bonus: float, divisor: float,
                 success_reward: float = 0.0, terminal_bonus: float = 0.0,
                 terminate_on_success: bool = False):
        super().__init__(env)
        self.success_bonus = float(success_bonus)
        self.divisor = float(divisor)
        self.success_reward = float(success_reward)
        self.terminal_bonus = float(terminal_bonus)
        self.terminate_on_success = bool(terminate_on_success)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        success = bool(info.get("sacflow_success", info.get("success", False)))
        if success and self.success_bonus:
            reward = float(reward) - (1.0 - self.success_reward) * self.success_bonus
        if self.divisor != 1.0:
            reward = float(reward) / self.divisor
        if success and self.terminal_bonus:
            reward = float(reward) + self.terminal_bonus
        if success and not self.terminate_on_success:
            terminated = False
        return obs, reward, terminated, truncated, info


def apply_precision(choice: str) -> None:

    if choice == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    _EFFECTIVE_PRECISION.update({
        "precision_choice": choice,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    })


def _gated_wrapper(actor_kwargs, precision="high"):
    class stage1_gated_actor(PRISTINE_GATED_ACTOR):


        def __init__(self, *actor_args, **kwargs):
            apply_precision(precision)
            campaign_kwargs = {"hidden_dim": 256, "max_iter_fwd": 25,
                               "tol_fwd": 1e-5, "exact_backward": True}
            campaign_kwargs.update(actor_kwargs)
            merged = {**kwargs, **campaign_kwargs}
            super().__init__(*actor_args, **merged)
            _capture_effective_actor(self)

    return stage1_gated_actor


ACTOR_WRAPPERS = {"gated": _gated_wrapper}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def scalar_failure(tag, value, max_q_loss):
    if tag.startswith(("loss/", "stats/", "coeff/", "grad/", "solver/", "diag/")):
        if not math.isfinite(float(value)):
            return f"nonfinite {tag}: {value}"
    if tag == "loss/q_loss" and float(value) > max_q_loss:
        return f"{tag}={value} exceeds stop bound {max_q_loss}"
    return None


def eval_interval_for(total_timesteps: int, horizon: int, smoke: bool) -> int:

    if smoke:
        return min(1_024, total_timesteps)
    if total_timesteps > 150_000 or horizon >= 200:
        return 50_000
    return 25_000


def actor_constructor(actor: str, args, actor_kwargs: dict | None = None) -> dict:
    if actor in ("gaussian", "parent", "dime"):


        return {"denoising_steps": args.denoising_steps}
    tol_fwd = float((actor_kwargs or {}).get("tol_fwd", 1e-5))
    if actor == "gated":
        return {"hidden_dim": 256, "max_iter_fwd": 25, "tol_fwd": tol_fwd,
                "exact_backward": True, "denoising_steps": args.denoising_steps,
                "integration": args.integration, "density_estimator": args.density_estimator,
                "gate_bias": args.gate_bias, "u_scale": args.u_scale,
                "jac_sigma_target": args.jac_sigma_target, "lam_fwd": 1e-2,
                "beta_fwd": 0.7, "m_fwd": 5, "max_iter_bwd": 10, "tol_bwd": 1e-3,
                "lam_bwd": 1e-2, "beta_bwd": 0.7, "m_bwd": 5}
    raise ValueError(f"Unsupported paper actor: {actor}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=tuple(RECIPES))
    parser.add_argument("environment", choices=RUN_ENVIRONMENTS)
    parser.set_defaults(seed=1)
    parser.add_argument("--run-id")
    parser.add_argument("--protocol", choices=("continuing", "soft"))
    parser.set_defaults(construction_seed_base=0)
    parser.add_argument("--actor", choices=tuple(ACTOR_WRAPPERS), default=None,
                        help="defaults to the recipe actor")
    parser.add_argument("--precision", choices=("high", "highest"), default=None,
                        help="floating-point matrix multiplication precision")
    parser.add_argument("--alpha-min", type=float, default=None)
    parser.add_argument("--alpha-leak", type=float, default=None)
    parser.add_argument("--lp-winsor", type=float, default=None)
    parser.add_argument("--ent-track", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--lam-jac", type=float, default=None)
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--anneal-steps", type=int, default=None)
    parser.add_argument("--tol-fwd", type=float, default=None)
    parser.add_argument("--no-tb", action=argparse.BooleanOptionalAction, default=None,
                        help="null TensorBoard writer with the scalar guard preserved")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--max-seconds", type=int, default=25_200)
    parser.add_argument("--resume-ckpt", type=Path, default=None,
                        help="seam-resume: warm-start actor/q1/q2/log_alpha from this "
                             "checkpoint and continue global_step up to --total-timesteps")
    parser.add_argument("--max-q-loss", type=float, default=1e12)
    parser.add_argument("--host", default="unknown")
    parser.add_argument("--gpu", default="unknown")
    parser.add_argument("--smoke", action="store_true",
                        help="disposable 2k integration run with both eval channels")
    parser.add_argument("--probe", action="store_true",
                        help="25k SPS probe (role marker; not a campaign cell)")
    parser.add_argument("--allow-steps-override", action="store_true",
                        help="permit total_timesteps to differ from 25k multiples bookkeeping")
    cli = parser.parse_args()

    actor = cli.actor or RECIPE_ACTOR[cli.recipe]
    method = RECIPE_METHOD.get(cli.recipe, "deflow")
    total_timesteps = cli.total_timesteps
    if total_timesteps is None:
        total_timesteps = SMOKE_STEPS if cli.smoke else 25_000 if cli.probe else None
    if total_timesteps is None:
        parser.error("--total-timesteps is required for a campaign run")
    cli.run_id = cli.run_id or f"{method}_{cli.environment}_{uuid.uuid4().hex[:12]}"
    if (Path(cli.run_id).name != cli.run_id or cli.run_id in (".", "..")
            or not cli.run_id
            or total_timesteps <= 0 or cli.max_seconds <= 0
            or not math.isfinite(cli.max_q_loss) or cli.max_q_loss <= 0):
        parser.error("require a simple run ID, positive steps "
                     "and finite bounds")
    bounds = (("alpha-min", cli.alpha_min, lambda value: value >= 0),
              ("alpha-leak", cli.alpha_leak, lambda value: value >= 0),
              ("lp-winsor", cli.lp_winsor, lambda value: 0.0 <= value < 0.5),
              ("alpha-init", cli.alpha_init, lambda value: value > 0),
              ("lam-jac", cli.lam_jac, lambda value: value >= 0),
              ("denoising-steps", cli.denoising_steps, lambda value: value >= 1),
              ("anneal-steps", cli.anneal_steps, lambda value: value >= 1),
              ("tol-fwd", cli.tol_fwd, lambda value: value > 0))
    for name, value, check in bounds:
        if value is not None and not check(value):
            parser.error(f"--{name}={value} is out of range")
    if not cli.smoke and total_timesteps % CHUNK:
        parser.error(f"total_timesteps must be a 25k multiple; got {total_timesteps}")
    protocol = cli.protocol or ("soft" if cli.environment.startswith("mw_") else "continuing")
    is_stage3 = protocol == "soft"
    if is_stage3:
        if cli.environment not in STAGE3_ENV_RULES:
            parser.error(f"{cli.environment} does not support the soft protocol")
        spec, rules = STAGE3_SPECS[cli.environment], STAGE3_ENV_RULES[cli.environment]
    else:
        if cli.environment not in ENV_RULES:
            parser.error(f"{cli.environment} does not support the continuing protocol")
        spec, rules = campaign.ENVIRONMENTS[cli.environment], ENV_RULES[cli.environment]
    __import__(spec["module"])
    divisor = rules["divisor"] if is_stage3 or RECIPE_NORMALIZED[cli.recipe] else 1.0
    horizon = rules["horizon"]
    spec_horizon = gym.spec(spec["env_id"]).max_episode_steps
    if spec_horizon is not None and int(spec_horizon) != horizon:
        raise RuntimeError(
            f"{spec['env_id']} horizon {spec_horizon} disagrees with the frozen table {horizon}")
    ckpt_interval = min(SMOKE_STEPS if cli.smoke else CHUNK, total_timesteps)
    if total_timesteps % ckpt_interval:
        parser.error(f"total_timesteps must be a multiple of the checkpoint grid {ckpt_interval}")

    run_dir = Path("runs") / cli.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")

    args = campaign.configure_method(method)
    if cli.resume_ckpt is not None:
        if not cli.resume_ckpt.is_file():
            parser.error(f"--resume-ckpt not found: {cli.resume_ckpt}")
        args.resume_ckpt = str(cli.resume_ckpt.resolve())
    args.exp_name = args.run_name = cli.run_id
    args.env_id, args.seed = spec["env_id"], cli.seed
    args.num_envs = 8
    args.total_timesteps = total_timesteps
    args.eval_envs = 5
    args.save_model = True
    args.best_reward_threshold_for_success = (
        spec["success_threshold"] if is_stage3 else
        (0.5 if RECIPE_NORMALIZED[cli.recipe] else spec["success_threshold"]))
    for key, value in RECIPES[cli.recipe].items():
        setattr(args, key, value)

    for key, value in (("alpha_min", cli.alpha_min), ("alpha_leak", cli.alpha_leak),
                       ("lp_winsor", cli.lp_winsor), ("lam_jac", cli.lam_jac),
                       ("denoising_steps", cli.denoising_steps),
                       ("anneal_steps", cli.anneal_steps)):
        if value is not None:
            setattr(args, key, value)
    if cli.ent_track is not None:
        args.ent_track = cli.ent_track
    if cli.alpha_init is not None:
        args.alpha_init = cli.alpha_init
    if cli.smoke:
        args.learning_starts = min(1_024, total_timesteps - 8)
        args.eval_interval = eval_interval_for(total_timesteps, horizon, True)
        args.eval_stochastic = 2
        args.ckpt_interval = ckpt_interval
        args.log_interval, args.tb_interval = 512, 64
    else:
        args.eval_interval = eval_interval_for(total_timesteps, horizon, False)
        args.eval_stochastic = 0
        args.ckpt_interval = ckpt_interval
    setattr(args, "stage1_recipe", cli.recipe)
    setattr(args, "stage1_actor", actor)
    setattr(args, "stage1_environment", cli.environment)
    setattr(args, "stage1_divisor", divisor)

    precision = cli.precision or RECIPE_PRECISION.get(cli.recipe, "high")
    no_tb = RECIPE_NO_TB.get(cli.recipe, False) if cli.no_tb is None else cli.no_tb
    success_reward = (rules.get("success_reward", 0.0) if is_stage3
                      else RECIPE_SUCCESS_REWARD.get(cli.recipe, 0.0))
    terminal_bonus = float(rules.get("terminal_bonus", 0.0)) if is_stage3 else 0.0
    terminate_on_success = (bool(rules.get("terminate_on_success", False))
                            if is_stage3 else False)
    recipe_actor_kwargs = dict(RECIPE_ACTOR_KWARGS.get(cli.recipe, {}))
    if cli.tol_fwd is not None:
        recipe_actor_kwargs["tol_fwd"] = cli.tol_fwd


    wrapper = None
    if method == "deflow":
        wrapper = ACTOR_WRAPPERS[actor](recipe_actor_kwargs, precision)
        base.deq_multistep_flow_actor = wrapper
    envs, writers = [], []

    construction_seed = bool(spec.get("construction_seed", False))
    construction_seed_base = cli.construction_seed_base

    def make_vec_env(env_id, seed, num_envs, capture_video, run_name):
        def thunk(index):
            def make():


                env = (gym.make(env_id, construction_seed=construction_seed_base + seed + index)
                       if construction_seed else gym.make(env_id))
                env = Stage1Reward(env, rules["success_bonus"], divisor,
                                   success_reward=success_reward,
                                   terminal_bonus=terminal_bonus,
                                   terminate_on_success=terminate_on_success)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.action_space.seed(seed + index)
                env.observation_space.seed(seed + index)
                return env
            return make
        env = gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])
        envs.append(env)
        return env

    original_writer = base.SummaryWriter

    class DiagnosticWriter(original_writer):
        def __init__(self, *writer_args, **writer_kwargs):
            super().__init__(*writer_args, **writer_kwargs)
            writers.append(self)

        def add_scalar(self, tag, scalar_value, global_step=None, *extra, **kwargs):
            super().add_scalar(tag, scalar_value, global_step, *extra, **kwargs)


            if (tag == "coeff/alpha" and getattr(args, "target_entropy_scale_final", 0.0) > 0
                    and envs and global_step is not None):
                anneal_steps = getattr(args, "anneal_steps", 0) or total_timesteps
                fraction = min(1.0, float(global_step) / float(anneal_steps))
                scale = (float(args.target_entropy_scale)
                         + (float(args.target_entropy_scale_final)
                            - float(args.target_entropy_scale)) * fraction)
                act_dim = math.prod(envs[0].single_action_space.shape)
                super().add_scalar("coeff/target_entropy_scheduled",
                                   -float(act_dim) * scale, global_step)
            failure = scalar_failure(tag, scalar_value, cli.max_q_loss)
            if failure:
                self.flush()
                raise FloatingPointError(f"step {global_step}: {failure}")

    def timeout(signum, frame):
        raise TimeoutError(f"stage1 run exceeded {cli.max_seconds} seconds")

    class NullWriter:


        def __init__(self, *writer_args, **writer_kwargs):
            writers.append(self)

        def add_scalar(self, tag, scalar_value, global_step=None, *extra, **kwargs):
            failure = scalar_failure(tag, scalar_value, cli.max_q_loss)
            if failure:
                raise FloatingPointError(f"step {global_step}: {failure}")

        def flush(self):
            pass

        def close(self):
            pass


    base.make_vec_env = make_vec_env
    base.SummaryWriter = NullWriter if no_tb else DiagnosticWriter

    evaluator_path = HERE.parent / "analysis" / "evaluate_stage1_panel.py"
    contract_family = protocol
    schema = "manipulation_v1"
    if is_stage3:
        reward_contract = (
            f"r_wrapped = (r_env - (1 - b)*1[success]) / {divisor} "
            f"+ {terminal_bonus}*1[success], b={success_reward}; "
            f"terminate_on_success={terminate_on_success}")
    elif RECIPE_NORMALIZED[cli.recipe]:
        reward_contract = (
            f"r_wrapped = (r_env - (1 - b)*1[success]) / {divisor}, b={success_reward}; "
            f"r_env includes the MS +{rules['success_bonus']} success bonus; "
            f"success termination suppressed")
    else:
        reward_contract = (
            f"r_wrapped = r_env - (1 - b)*1[success], b={success_reward}; no divisor")
    metadata = {
        "schema": schema,
        "contract_schema_version": 3 if is_stage3 else 2,
        "continued": bool(cli.resume_ckpt),
        "resume_ckpt": (str(cli.resume_ckpt.resolve()) if cli.resume_ckpt else None),
        "scientific_role": ("integration_smoke" if cli.smoke else
                            "sps_probe" if cli.probe else
                            "training"),
        "run_id": cli.run_id, "environment": cli.environment, "env_id": spec["env_id"],
        "recipe": cli.recipe, "actor": actor, "method": method, "seed": cli.seed,
        "protocol": protocol,
        "construction_seed_base": construction_seed_base,
        "precision": precision,
        "effective_precision": dict(_EFFECTIVE_PRECISION),
        "tb_writer": "null" if no_tb else "tensorboard",
        "total_timesteps": total_timesteps, "horizon": horizon,
        "host": cli.host, "gpu": cli.gpu,
        "ckpt_interval": ckpt_interval, "eval_interval": args.eval_interval,
        "in_training_stochastic_eval": args.eval_stochastic,
        "max_seconds": cli.max_seconds, "max_q_loss": cli.max_q_loss,
        "recipe_constants": RECIPES[cli.recipe],
        "recipe_normalized": RECIPE_NORMALIZED[cli.recipe],
        "divisor": divisor, "success_bonus": rules["success_bonus"],
        "success_reward_b": success_reward,
        "terminal_bonus": terminal_bonus,
        "terminate_on_success": terminate_on_success,
        "soft_protocol": ({
            "family": ("metaworld_soft" if cli.environment.startswith("mw_")
                       else "maniskill_soft"),
            "horizon": horizon,
            "divisor": divisor,
            "terminal_bonus": terminal_bonus,
            "terminate_on_success": terminate_on_success,
        } if is_stage3 else None),
        "entropy_schedule": ({
            "initial_scale": args.target_entropy_scale,
            "final_scale": args.target_entropy_scale_final,
            "anneal_steps": args.anneal_steps or total_timesteps,
            "tag": "coeff/target_entropy",
        } if args.target_entropy_scale_final > 0 else None),
        "reward_contract": reward_contract,
        "determinism_env": {name: os.environ.get(name) for name in DETERMINISM_ENV_VARS},
        "torch_determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        },
        "actor_constructor": actor_constructor(actor, args, recipe_actor_kwargs),
        "effective_actor": dict(_EFFECTIVE_ACTOR),
        "actor_class": (f"run_manipulation_stage1.{wrapper.__name__}"
                        if wrapper is not None else
                        f"{base.deq_multistep_flow_actor.__module__}."
                        f"{base.deq_multistep_flow_actor.__qualname__}"),
        "actor_module_sha256": file_sha256(
            (sys.modules["sac_gaussian_baseline"].__file__ if actor == "gaussian" else
             sys.modules["sac_flow_parent"].__file__ if actor == "parent" else
             sys.modules["dime_actor"].__file__ if actor == "dime" else base.__file__)),
        "runner_module_sha256": file_sha256(__file__),
        "engine_module_sha256": file_sha256(base.__file__),
        "campaign_module_sha256": file_sha256(campaign.__file__),
        "maniskill_env_module_sha256": file_sha256(HERE.parent / "maniskill_env.py"),
        "evaluator_module_sha256": (file_sha256(evaluator_path)
                                    if evaluator_path.is_file() else None),
        "configuration": json_safe(asdict(args)),

    }
    contract_path = run_dir / f"{contract_family}_{actor}_{cli.recipe}_contract.json"
    campaign.atomic_json(contract_path, metadata)
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cli.max_seconds)
    started = time.perf_counter()
    try:
        base.main(args)
        checkpoint = run_dir / f"ckpt_step_{total_timesteps:09d}.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"training returned without final checkpoint: {checkpoint}")
        wall_seconds = time.perf_counter() - started
        completion = {
            **metadata, "status": "training_complete", "global_step": total_timesteps,
            "wall_seconds": wall_seconds,
            "env_steps_per_second": total_timesteps / max(wall_seconds, 1e-6),
            "effective_actor": dict(_EFFECTIVE_ACTOR),
            "effective_precision": dict(_EFFECTIVE_PRECISION),
            "final_checkpoint": str(checkpoint.resolve()),
            "final_checkpoint_sha256": file_sha256(checkpoint),

        }
        campaign.atomic_json(contract_path, completion)
        campaign.atomic_json(run_dir / "training_complete.json",
                             {**completion, "status": "training_complete"})
        campaign.atomic_json(run_dir / "SUCCESS.json",
                             {**completion, "status": "succeeded"})
    except Exception as exc:
        campaign.atomic_json(run_dir / "FAILURE.json", {
            **metadata, "status": "failed_stage1_run", "error_type": type(exc).__name__,
            "error": str(exc),
            "effective_actor": dict(_EFFECTIVE_ACTOR),
            "effective_precision": dict(_EFFECTIVE_PRECISION),
        })
        raise
    finally:
        signal.alarm(0)
        for env in envs:
            env.close()
        for writer in writers:
            writer.close()


if __name__ == "__main__":
    main()
