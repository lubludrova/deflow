"""Shared method setup for the paper manipulation recipes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC))

import sac_anderson_flow as base


REFERENCE_DEFLOW_ACTOR = base.deq_multistep_flow_actor


ENVIRONMENTS = {
    "ms_pickcube": {
        "env_id": "MS3PickCube-v0",
        "run_tag": "ms_pickcube",
        "success_threshold": 0.5,
        "module": "maniskill_env",
    },
    "ms_pushcube": {
        "env_id": "MS3PushCube-v0",
        "run_tag": "ms_pushcube",
        "success_threshold": 0.5,
        "module": "maniskill_env",
    },
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configure_method(method: str):
    args = base.Args()
    if method == "sac":
        import sac_gaussian_baseline as gaussian
        import sac_flow_parent as parent

        base.deq_multistep_flow_actor = gaussian.gaussian_actor
        base.critic = parent.parent_critic
        args.target_entropy_scale = 1.0
    elif method == "sacflow":
        import sac_flow_parent as parent

        base.deq_multistep_flow_actor = parent.parent_flow_actor
        base.critic = parent.parent_critic
        args.alpha_init = 0.2
        args.target_entropy_scale = 0.0
    elif method == "dime":
        import dime_actor as dime
        import sac_flow_parent as parent

        base.deq_multistep_flow_actor = dime.dime_dis_actor
        base.critic = parent.parent_critic
        args.alpha_init = 1.0
        args.target_entropy_scale = 4.0 - (
            0.5 * math.log(2 * math.pi * math.e) + math.log(dime.PRIOR_STD)
        )
        args.denoising_steps = 16
        args.alpha_min = 0.0

        args.logp_clip = 0.0
        args.grad_clip = 0.0
    else:
        import sac_flow_parent as parent

        class deflow_actor(REFERENCE_DEFLOW_ACTOR):
            def __init__(self, *actor_args, **actor_kwargs):
                actor_kwargs["hidden_dim"] = 256
                actor_kwargs["max_iter_fwd"] = 25
                actor_kwargs["tol_fwd"] = 3e-4
                actor_kwargs["exact_backward"] = True
                super().__init__(*actor_args, **actor_kwargs)

        base.deq_multistep_flow_actor = deflow_actor
        base.critic = parent.parent_critic
        args.u_scale = 4.0
        args.gate_bias = 1.0
        args.lam_jac = 1e-2
        args.jac_sigma_target = 0.9
        args.jac_warmup = 100_000
        args.ent_track = True
        args.ent_margin = 0.7
        args.ent_floor_scale = 0.7
        args.alpha_leak = 1e-2
        args.alpha_prior = 0.05
        args.lp_winsor = 0.05
        args.logp_clip = 15.0
        args.grad_clip = 1.0
        args.alpha_min = 0.02

    return args
