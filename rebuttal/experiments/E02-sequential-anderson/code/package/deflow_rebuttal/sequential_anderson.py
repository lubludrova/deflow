"""Sequential implicit Euler using the unchanged native Anderson kernel.

Every local equation uses previous + dt * field(x, obs, t), starts at previous,
and has a fresh history. Tolerance is the native batch-mean relative criterion
at each step. It is not a per-sample accuracy guarantee. The caller independently
reevaluates local residuals and attaches the same implicit backward as E01.
"""
import math

import torch


@torch.no_grad()
def solve_sequential_anderson(field, obs, z, *, steps, anderson, dt=None,
                              m=5, lam=1e-2, max_iter=25, tol=1e-5, beta=0.7):
    if z.ndim != 2 or obs.ndim != 2 or len(obs) != len(z) or not z.numel() or steps < 1:
        raise ValueError('Expected nonempty z[B,d], obs[B,o], and positive steps')
    if z.dtype not in (torch.float32, torch.float64) or z.dtype != obs.dtype or z.device != obs.device:
        raise ValueError('obs and z must share FP32/FP64 dtype and device')
    dt = 1.0 / steps if dt is None else float(dt)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(tol) or tol <= 0 or max_iter < 3 or m < 2:
        raise ValueError('Invalid Anderson settings')
    previous = z.detach().clone()
    trajectory, records = [], []
    evaluations = 0
    for k in range(steps):
        times = torch.full((len(z), 1), (k + 1) * dt, dtype=z.dtype, device=z.device)

        def mapping(x):
            nonlocal evaluations
            evaluations += len(x)
            return previous + dt * field(x, obs, times)

        result, stats = anderson(mapping, previous.clone(), m=m, lam=lam,
                                 max_iter=max_iter, tol=tol, beta=beta, return_stats=True)
        residual = stats['res_last']
        converged = residual is not None and math.isfinite(residual) and residual < tol
        limit = stats['n_iter'] >= max_iter
        records.append(dict(stats, mean_criterion_met=converged,
                            iteration_limit_reached=limit,
                            budget_exhausted_without_convergence=limit and not converged))
        trajectory.append(result)
        previous = result
    native = dict(
        n_iter=max(s['n_iter'] for s in records),
        iterations_sum=sum(s['n_iter'] for s in records),
        cap_hit=any(s['iteration_limit_reached'] for s in records),
        mean_criterion_met=all(s['mean_criterion_met'] for s in records),
        budget_exhausted_without_convergence=any(s['budget_exhausted_without_convergence'] for s in records),
        res_last=max(s['res_last'] for s in records),
        convergence_scope='native batch mean at every local step, not all samples',
        tolerance=tol, iteration_limit=max_iter,
        anderson_history_lstsq_count=sum(max(0, s['n_iter'] - 2) for s in records),
        per_step=records)
    raw = dict(nfe_total=evaluations, jacobian_evals_total=0, linear_solves_total=0,
               anderson_history_lstsq_total=len(z)*native['anderson_history_lstsq_count'],
               per_step=records)
    return torch.stack(trajectory, dim=1), native, raw
