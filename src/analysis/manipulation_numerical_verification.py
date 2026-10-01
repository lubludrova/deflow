"""Numerical reference helpers for trained-policy verification."""

import hashlib
import importlib.util
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacrev, vmap


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime():
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "threads": torch.get_num_threads(), "device": "cpu"}


def jacobians(actor, obs, seq):
    batch, steps, dim = seq.shape
    times = (torch.arange(1, steps + 1, dtype=seq.dtype) * actor.dt).view(1, steps, 1)
    def field(u, o, t):
        return actor.vf(u[None], o[None], t[None])[0]
    return vmap(jacrev(field, argnums=0))(
        seq.flatten(0, 1), obs[:, None].expand(-1, steps, -1).flatten(0, 1),
        times.expand(batch, -1, -1).flatten(0, 1)
    ).reshape(batch, steps, dim, dim)


def density(actor, obs, z, seq):
    j = jacobians(actor, obs, seq)
    eye = torch.eye(actor.act_dim, dtype=seq.dtype)
    matrix = eye - actor.dt * j
    raw_ld = torch.linalg.slogdet(matrix)[1]
    shifted_ld = torch.linalg.slogdet(matrix + actor.logdet_eps * eye)[1]
    endpoint = seq[:, -1]
    sech2 = 1 - endpoint.tanh().square()

    stable_log_sech2 = 2 * (np.log(2.) - endpoint - torch.nn.functional.softplus(-2 * endpoint))
    base = -.5 * (z.square().sum(-1) / actor.z_std ** 2
                  + actor.act_dim * np.log(2 * np.pi * actor.z_std ** 2))
    raw = base + raw_ld.sum(-1) - (stable_log_sech2 + actor.action_scale.abs().log()).sum(-1)
    guarded = base + shifted_ld.clamp(-50, 50).sum(-1) - (
        sech2.clamp_min(1e-6).log() + (actor.action_scale.abs() + 1e-6).log()).sum(-1)
    singular = torch.linalg.svdvals(matrix)
    return {"raw": raw, "guarded": guarded, "j": j,
            "sigma_min": singular[..., -1], "condition": singular[..., 0] / singular[..., -1],
            "det_clip": shifted_ld.abs() > 50, "squash_floor": sech2 <= 1e-6,
            "shift_bias": (shifted_ld - raw_ld).sum(-1)}


def solve(engine, actor, obs, z, cap):
    a0 = actor._make_a0_from_z(z)
    started = time.perf_counter()
    with torch.no_grad():
        seq, telemetry = engine.anderson(lambda x: actor._g(x, a0, obs), a0.clone(),
            m=actor.m_fwd, lam=actor.lam_fwd, max_iter=cap,
            tol=actor.tol_fwd, beta=actor.beta_fwd, return_stats=True)
        attached = actor._g(seq, a0, obs)
    elapsed = time.perf_counter() - started
    residual = (attached - seq).norm(dim=1) / (attached.norm(dim=1) + 1e-6)
    return seq.reshape(-1, actor.T, actor.act_dim), attached.reshape(-1, actor.T, actor.act_dim), residual, telemetry, elapsed


def reference_root(actor, obs, z, initial):
    from scipy.optimize import root
    previous = z.numpy().copy()
    seq, residuals = [], []
    for k in range(actor.T):
        t = torch.tensor([[(k + 1) * actor.dt]], dtype=torch.float64)
        def equation(x):
            with torch.no_grad():
                v = actor.vf(torch.from_numpy(x)[None], obs[None], t)[0].numpy()
            return x - actor.dt * v - previous
        def derivative(x):
            def field(u):
                return actor.vf(u[None], obs[None], t)[0]
            return np.eye(actor.act_dim) - actor.dt * jacrev(field)(torch.from_numpy(x)).detach().numpy()
        answer = root(equation, initial[k].numpy(), jac=derivative, method="hybr", tol=1e-11)
        residuals.append(float(np.linalg.norm(equation(answer.x))))
        seq.append(answer.x.copy())
        previous = answer.x.copy()
    return torch.tensor(np.stack(seq)), residuals


def joint_matrix(actor, j):
    batch, steps, dim, _ = j.shape
    block = torch.stack([torch.block_diag(*sample.unbind()) for sample in j])
    return torch.eye(steps * dim, dtype=j.dtype)[None] - actor.dt * (actor.matrix_in_g[None] @ block)


def critics(payload, dtype):
    models = []
    for name in ("q1", "q2"):
        state = payload[name]
        indices = sorted(int(k.split(".")[1]) for k in state if k.endswith("weight"))
        layers = []
        for i in indices:
            weight = state[f"net.{i}.weight"]
            layers.append(torch.nn.Linear(weight.shape[1], weight.shape[0]))
            if i != indices[-1]:
                layers.append(torch.nn.ReLU())
        net = torch.nn.Sequential(*layers).to(dtype=dtype)
        net.load_state_dict({k.removeprefix("net."): v for k, v in state.items()})
        net.requires_grad_(False)
        models.append(net)
    return models
