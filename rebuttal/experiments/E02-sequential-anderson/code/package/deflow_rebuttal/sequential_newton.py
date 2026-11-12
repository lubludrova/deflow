"""Sequential implicit Euler with batched exact Newton and Armijo backtracking.

Each row is independent. The field must not couple batch rows (DEFlow's
per-example LayerNorm is compatible). This is a forward root finder, not a
finite-program gradient implementation. Attach implicit differentiation outside.

Merit phi = ||F||^2 / 2, F = x - previous - dt * field(x). Newton solves
J_F p = -F. Armijo accepts phi(x + alpha*p) <= phi(x) + c*alpha*F^T J_F p.
No Jacobian shift, clipping, least-squares rescue, or hidden solver fallback is
used. Failure keeps the last accepted iterate and is explicitly reported.

References: Kelley, Iterative Methods for Linear and Nonlinear Equations,
SIAM 1995, chapter 8: https://doi.org/10.1137/1.9781611970944.ch8
SciPy v1.14.1 optimize/_nonlin.py:
https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_nonlin.py

_nonlin_line_search, uses a squared-residual merit and Armijo globalization.
Unlike that reference implementation, failed line searches do not take an
unverified full step. Code here is an independent batched implementation.
"""
from dataclasses import dataclass
import math
import torch
from torch.func import jacrev, vmap

STATUS_NAMES = {0: "converged", 1: "max_iter", 2: "nonfinite", 3: "singular",
                4: "line_search_failed", 5: "non_descent"}


@dataclass(frozen=True)
class NewtonConfig:
    rtol: float = 1e-5
    atol: float = 1e-7
    max_iter: int = 25
    max_backtracks: int = 12
    armijo: float = 1e-4
    shrink: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.rtol, self.atol, self.armijo, self.shrink)):
            raise ValueError("Solver settings must be finite")
        if self.rtol < 0 or self.atol < 0 or self.rtol + self.atol == 0:
            raise ValueError("At least one nonnegative residual tolerance must be positive")
        if self.max_iter < 1 or self.max_backtracks < 0:
            raise ValueError("Invalid iteration/backtracking limits")
        if not (0 < self.armijo < 0.5 and 0 < self.shrink < 1):
            raise ValueError("Invalid Armijo or backtracking factor")


def local_residual_metrics(x, previous, residual):
    """Per-row residuals with one documented scale for both solver audits."""
    absolute = torch.linalg.vector_norm(residual, dim=-1)
    scale = torch.maximum(torch.linalg.vector_norm(x, dim=-1),
                          torch.linalg.vector_norm(previous, dim=-1))
    relative = absolute / scale.clamp_min(torch.finfo(x.dtype).tiny)
    return absolute, relative, scale


@torch.no_grad()
def solve_sequential(field, obs, z, *, steps, dt=None, config=None):
    """Return detached trajectory[B,T,d] and per-row/per-step diagnostics.

    Initial guess for every step is the previous action. Status is local to
    each step, so all_converged is the AND across steps. A failed preceding step
    remains visible even when later local equations converge. nfe counts field
    positions, including one primal field evaluation per Jacobian evaluation.
    jacobian_evals counts the extra autodiff operation separately. CUDA timings
    must be synchronized by the caller. This function works inside no_grad,
    but not inference_mode (Jacobians require autograd metadata).
    """
    config = config or NewtonConfig()
    if z.ndim != 2 or obs.ndim != 2 or len(obs) != len(z) or steps < 1 or z.numel() == 0:
        raise ValueError("Expected z[B,d], obs[B,o], and positive steps")
    if z.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use FP32 or FP64 for the nonlinear solver")
    if z.device != obs.device or z.dtype != obs.dtype:
        raise ValueError("obs and z must have identical dtype/device")
    dt = 1.0 / steps if dt is None else float(dt)
    if not math.isfinite(dt) or not dt > 0:
        raise ValueError("dt must be positive")
    B, d = z.shape
    previous = z.detach().clone()
    trajectory, records = [], []
    eye = torch.eye(d, device=z.device, dtype=z.dtype)

    for k in range(steps):
        x = previous.clone()
        times = torch.full((B, 1), (k + 1) * dt, device=z.device, dtype=z.dtype)
        counts = {name: torch.zeros(B, dtype=torch.long, device=z.device)
                  for name in ("iterations", "nfe", "jacobian_evals", "linear_solves", "backtracks")}
        status = torch.full((B,), 1, dtype=torch.long, device=z.device)
        min_alpha = torch.ones(B, dtype=z.dtype, device=z.device)
        residual = x - previous - dt * field(x, obs, times)
        counts["nfe"] += 1

        def classify(ids):
            finite = torch.isfinite(residual[ids]).all(-1) & torch.isfinite(x[ids]).all(-1)
            absolute, _, scale = local_residual_metrics(x[ids], previous[ids], residual[ids])
            converged = finite & (absolute <= config.atol + config.rtol * scale)
            status[ids[~finite]] = 2
            status[ids[converged]] = 0
            return ids[finite & ~converged]

        active = classify(torch.arange(B, device=z.device))
        for _ in range(config.max_iter):
            if active.numel() == 0:
                break
            counts["iterations"][active] += 1

            def single(a, o, t):
                return field(a[None], o[None], t[None])[0]

            with torch.enable_grad():
                jac = vmap(jacrev(single, argnums=0))(x[active], obs[active], times[active]).detach()
            counts["jacobian_evals"][active] += 1
            counts["nfe"][active] += 1
            matrix = eye[None] - dt * jac
            finite_matrix = torch.isfinite(matrix).all(dim=(-2, -1))
            status[active[~finite_matrix]] = 2
            active = active[finite_matrix]
            matrix = matrix[finite_matrix]
            if active.numel() == 0:
                break
            direction, info = torch.linalg.solve_ex(matrix, -residual[active].unsqueeze(-1), check_errors=False)
            direction = direction.squeeze(-1)
            counts["linear_solves"][active] += 1
            valid = (info == 0) & torch.isfinite(direction).all(-1)
            status[active[~valid]] = 3
            matrix, direction, active = matrix[valid], direction[valid], active[valid]
            if active.numel() == 0:
                break
            slope = (residual[active] * torch.bmm(matrix, direction.unsqueeze(-1)).squeeze(-1)).sum(-1)
            descending = torch.isfinite(slope) & (slope < 0)
            status[active[~descending]] = 5
            direction, slope, active = direction[descending], slope[descending], active[descending]
            if active.numel() == 0:
                break
            base = x[active].clone()
            phi = 0.5 * residual[active].square().sum(-1)
            alpha = torch.ones(len(active), dtype=z.dtype, device=z.device)
            pending = torch.arange(len(active), device=z.device)
            accepted = torch.zeros(len(active), dtype=torch.bool, device=z.device)
            for attempt in range(config.max_backtracks + 1):
                if pending.numel() == 0:
                    break
                ids = active[pending]
                candidate = base[pending] + alpha[pending, None] * direction[pending]
                trial = candidate - previous[ids] - dt * field(candidate, obs[ids], times[ids])
                counts["nfe"][ids] += 1
                if attempt:
                    counts["backtracks"][ids] += 1
                trial_phi = 0.5 * trial.square().sum(-1)
                good = (torch.isfinite(candidate).all(-1) & torch.isfinite(trial).all(-1)
                        & torch.isfinite(trial_phi)
                        & (trial_phi <= phi[pending] + config.armijo * alpha[pending] * slope[pending]))
                accepted_ids = ids[good]
                x[accepted_ids] = candidate[good]
                residual[accepted_ids] = trial[good]
                min_alpha[accepted_ids] = torch.minimum(min_alpha[accepted_ids], alpha[pending[good]])
                accepted[pending[good]] = True
                pending = pending[~good]
                alpha[pending] *= config.shrink
            status[active[~accepted]] = 4
            active = classify(active[accepted])

        absolute, relative, scale = local_residual_metrics(x, previous, residual)
        record = dict(counts, status=status, abs_residual=absolute,
                      rel_residual=relative, residual_scale=scale,
                      accepted_damping_min=min_alpha,
                      cap_hit=counts["iterations"] >= config.max_iter)
        records.append(record)
        trajectory.append(x)
        previous = x

    per_step = {key: torch.stack([item[key] for item in records], dim=1) for key in records[0]}
    stats = {"per_step": per_step, "all_converged": (per_step["status"] == 0).all(1),
             "status_names": STATUS_NAMES,
             "nfe_total": int(per_step["nfe"].sum().item()),
             "jacobian_evals_total": int(per_step["jacobian_evals"].sum().item()),
             "linear_solves_total": int(per_step["linear_solves"].sum().item())}
    return torch.stack(trajectory, dim=1), stats
