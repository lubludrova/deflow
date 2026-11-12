"""Frozen full-batch numerical release gates. Never train or alter solver settings.

Run with revision3's interpreter. ROOT contains package/ and DEFlow/. Every
output directory is new. Verdict pass means implementation/policy gates passed,
not that every nonlinear equation or independent reference converged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.func import jacrev

POLICY = 'strict_fp32'
THRESHOLDS = {
    'rtol': 1e-5, 'atol': 1e-7, 'max_iter': 25,
    'boundary_uncertainty_fraction': 0.1, 'norm_rounding_eps_multiple': 8,
    'fd_steps': [1e-5, 1e-6, 1e-7], 'fd_relative': 1e-6, 'fd_absolute': 1e-7,
    'readout_atol': 1e-6, 'readout_rtol': 1e-4, 'gradient_relative': 1e-3,
}
FRESH_ARMS = {'newton', 'anderson', 'anderson_highcap', 'anderson_tighter'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(k): cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def finite(*values):
    return all(bool(torch.isfinite(x).all()) for x in values)


def write_json(path, value):
    from deflow_rebuttal.audit import clean
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def tolerance(q, z):
    previous = torch.cat((z[:, None], q[:, :-1]), dim=1)
    scale = torch.maximum(q.norm(dim=-1), previous.norm(dim=-1))
    return THRESHOLDS['atol'] + THRESHOLDS['rtol'] * scale


@torch.no_grad()
def residual_layouts(actor, obs, z, q):
    """Reevaluate fixed q. No re-solving and no selection of favorable rows."""
    batch, steps, dim = q.shape
    previous = torch.cat((z[:, None], q[:, :-1]), dim=1)
    flat = q.flatten(0, 1)
    states = obs[:, None].expand(-1, steps, -1).flatten(0, 1)
    times = (torch.arange(1, steps + 1, device=q.device, dtype=q.dtype) * actor.dt)
    times = times[None, :, None].expand(batch, -1, -1).flatten(0, 1)

    def evaluate(order, chunk):
        values = torch.empty_like(flat)
        for ids in order.split(chunk):
            values[ids] = actor.vf(flat[ids], states[ids], times[ids])
        return q - previous - actor.dt * values.reshape_as(q)

    order = torch.arange(len(flat), device=q.device)
    result = {'flattened': evaluate(order, len(flat))}
    # Reverse order is fixed in advance and covers every local equation.
    result['permuted'] = evaluate(order.flip(0), len(flat))
    for chunk in (1, 32, 1024):
        result[f'chunk_{chunk}'] = evaluate(order, chunk)
    per_step = []
    for k in range(steps):
        t = torch.full((batch, 1), (k + 1) * actor.dt, device=q.device, dtype=q.dtype)
        per_step.append(q[:, k] - previous[:, k] - actor.dt * actor.vf(q[:, k], obs, t))
    result['per_step'] = torch.stack(per_step, dim=1)
    return result


def classify_boundary(native_abs, native_status, residuals, residual64, q, z):
    """Keep exact masks. Explained rounding is a separate diagnostic category."""
    tau = tolerance(q, z).double()
    tau64 = tolerance(q.double(), z.double())
    norms = {name: value.norm(dim=-1).double() for name, value in residuals.items()}
    norms['stored_native'] = native_abs.double()
    reference_abs = residual64.norm(dim=-1)
    masks = {name: value <= tau for name, value in norms.items()}
    masks['fp64_fixed_q'] = reference_abs <= tau64
    native_ok = native_status == 0
    disagreement = torch.zeros_like(native_ok)
    for mask in masks.values():
        disagreement |= mask != native_ok
    # Actual cross-layout and FP64 discrepancies at fixed q, with only a norm
    # arithmetic allowance. Keep the state-scale allowance visible separately.
    norm_allowance = THRESHOLDS['norm_rounding_eps_multiple'] * torch.finfo(torch.float32).eps
    measured = torch.stack([(value - reference_abs).abs() for value in norms.values()]).amax(0)
    norm_allowance = norm_allowance * torch.stack(list(norms.values()) + [reference_abs]).amax(0)
    uncertainty = measured + norm_allowance + (tau - tau64).abs()
    margin = (reference_abs - tau64).abs()
    explained = disagreement & native_ok & (margin <= uncertainty) & (
        uncertainty <= THRESHOLDS['boundary_uncertainty_fraction'] * tau64)
    unexplained = disagreement & native_ok & ~explained
    return dict(tolerance=tau, tolerance_fp64=tau64, absolute=norms,
                fp64_absolute=reference_abs, exact_pass_masks=masks,
                normalized_margin=(reference_abs - tau64) / tau64,
                measured_uncertainty=measured, norm_allowance=norm_allowance,
                total_uncertainty=uncertainty, uncertainty_over_tolerance=uncertainty / tau64,
                disagreement=disagreement, explained_boundary=explained,
                unexplained_native_success=unexplained)


def finite_difference_gate(actor, obs, seq, audit):
    """All B*T two-dimensional Jacobians, batched central differences in FP64."""
    q = seq.detach()
    batch, steps, dim = q.shape
    analytic = audit.jacobians(actor, obs, q).detach()
    states = obs[:, None].expand(-1, steps, -1).flatten(0, 1)
    times = (torch.arange(1, steps + 1, device=q.device, dtype=q.dtype) * actor.dt)
    times = times[None, :, None].expand(batch, -1, -1).flatten(0, 1)
    flat = q.flatten(0, 1)
    candidates, absolute, relative = [], [], []
    with torch.no_grad():
        for base_h in THRESHOLDS['fd_steps']:
            columns = []
            for coordinate in range(dim):
                h = base_h * flat[:, coordinate].abs().clamp_min(1)
                shift = torch.zeros_like(flat)
                shift[:, coordinate] = h
                derivative = (actor.vf(flat + shift, states, times) - actor.vf(flat - shift, states, times)) / (2 * h[:, None])
                columns.append(derivative)
            estimate = torch.stack(columns, -1).view_as(analytic)
            error = torch.linalg.matrix_norm(estimate - analytic)
            candidates.append(estimate)
            absolute.append(error)
            relative.append(error / torch.linalg.matrix_norm(analytic).clamp_min(1e-30))
    absolute, relative = torch.stack(absolute), torch.stack(relative)
    passed = ((absolute <= THRESHOLDS['fd_absolute']) | (relative <= THRESHOLDS['fd_relative'])).any(0)
    return dict(analytic=analytic, estimates=torch.stack(candidates), absolute_error=absolute,
                relative_error=relative, pass_mask=passed,
                pass_all=finite(analytic, absolute, relative) and bool(passed.all()))


def sac_loss(actor, obs, action, logp, critics, config):
    clipped = logp.clamp(config['logp_lower'], config['logp_upper'])
    qvalue = torch.minimum(*(net(torch.cat((obs, action), -1)) for net in critics)).squeeze(-1)
    loss = (config['alpha'] * clipped - qvalue).mean()
    if config['effective_lam_jac']:
        if actor._jac_pen is None:
            raise ValueError('Missing requested native Jacobian penalty')
        loss = loss + config['effective_lam_jac'] * actor._jac_pen.mean()
    return loss


def native_readout(engine, actor, obs, z, seq, critics, config):
    """Actual unmodified native attachment/backward at the identical frozen q."""
    from deflow_rebuttal import precision
    events = {}
    hooks = []
    original_solver = engine.anderson
    original_sample = actor._sample_base_z
    try:
        engine.anderson = lambda *args, **kwargs: (seq.detach().flatten(1), {'n_iter': 0, 'res_last': 0.0})
        actor._sample_base_z = lambda *args: z
        actor._jac_pen = None
        precision.assert_float32(POLICY, obs, seq, *actor.parameters())
        events['density_entry'] = precision.snapshot()
        action, logp = actor.forward_with_logprob(obs)
        precision.assert_float32(POLICY, action, logp)
        events['density_result'] = precision.snapshot()
        def observe_parameter_gradient(gradient):
            precision.assert_float32(POLICY, gradient)
            events.setdefault('native_parameter_gradient', precision.snapshot())
            return gradient
        for parameter in actor.parameters():
            if parameter.requires_grad:
                hooks.append(parameter.register_hook(observe_parameter_gradient))
        loss = sac_loss(actor, obs, action, logp, critics, config)
        parameters = list(actor.parameters())
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradient = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                              for p, g in zip(parameters, gradients)])
        return {'action': action.detach(), 'logp': logp.detach(), 'loss': loss.detach(),
                'gradient': gradient.detach(), 'density': copy.deepcopy(actor.last_diag),
                'attachment': copy.deepcopy(actor.last_fwd), 'runtime_boundaries': events}
    finally:
        for hook in hooks:
            hook.remove()
        engine.anderson = original_solver
        actor._sample_base_z = original_sample


def parity_gate(engine, actor, obs, z, seq, payload, audit):
    config = dict(payload['loss_config'])
    for name, default in [('logp_lower', -float('inf')), ('logp_upper', float('inf'))]:
        if config[name] is None:
            config[name] = default
    critics = audit.critics(payload, str(obs.device), obs.dtype)
    native = native_readout(engine, actor, obs, z, seq, critics, config)
    common = audit.readout(actor, obs, z, seq, critics, config)
    parity = all(torch.allclose(native[key], common[key], atol=THRESHOLDS['readout_atol'],
                               rtol=THRESHOLDS['readout_rtol']) for key in ('action', 'logp', 'loss'))
    gradient_relative = (native['gradient'] - common['gradient']).norm() / native['gradient'].norm().clamp_min(1e-12)
    values_finite = finite(*(native[key] for key in ('action', 'logp', 'loss', 'gradient')),
                           *(common[key] for key in ('action', 'logp', 'loss', 'gradient')))
    # Retain the actual common shifted and unshifted adjoint diagnostics. Their
    # RHS-relative residual can be amplified by conditioning and is not used as
    # an unqualified backward-error threshold.
    return dict(native=native, audited=common, adjoint=cpu(getattr(actor, '_audit_adjoint', {})),
                readout_close=parity, finite=values_finite,
                gradient_relative=float(gradient_relative),
                pass_all=values_finite and parity and float(gradient_relative) <= THRESHOLDS['gradient_relative'])


def trace_local(actor, obs, previous, step, config):
    """Independent scalar-row FP64 trace of the unchanged Newton/Armijo rules."""
    x = previous.detach().clone()
    t = torch.full((1, 1), (step + 1) * actor.dt, dtype=x.dtype, device=x.device)
    def equation(point):
        return point - previous - actor.dt * actor.vf(point[None], obs[None], t)[0]
    trace = []
    status = 'max_iter'
    for iteration in range(config.max_iter):
        residual = equation(x).detach()
        tau = config.atol + config.rtol * torch.maximum(x.norm(), previous.norm())
        if not finite(x, residual, tau):
            status = 'nonfinite'
            break
        if residual.norm() <= tau:
            status = 'converged'
            break
        matrix = jacrev(equation)(x).detach()
        p, info = torch.linalg.solve_ex(matrix, -residual)
        if int(info) or not finite(matrix, p):
            status = 'singular'
            break
        slope = residual @ (matrix @ p)
        entry = dict(iteration=iteration, x=x.detach(), residual=residual, tolerance=tau,
                     matrix=matrix, sigma=torch.linalg.svdvals(matrix), direction=p,
                     linear_backward_error=(matrix @ p + residual).norm() / (
                         matrix.norm() * p.norm() + residual.norm()).clamp_min(1e-30),
                     slope=slope, trials=[])
        trace.append(entry)
        if not torch.isfinite(slope) or slope >= 0:
            status = 'non_descent'
            break
        phi = residual.square().sum() / 2
        accepted = False
        for attempt in range(config.max_backtracks + 1):
            alpha = config.shrink ** attempt
            candidate = x + alpha * p
            trial = equation(candidate).detach()
            trial_phi = trial.square().sum() / 2
            bound = phi + config.armijo * alpha * slope
            good = finite(candidate, trial, trial_phi) and bool(trial_phi <= bound)
            entry['trials'].append(dict(alpha=alpha, candidate=candidate.detach(), residual=trial,
                                        phi=trial_phi, armijo_bound=bound, accepted=good))
            if good:
                x = candidate.detach()
                accepted = True
                break
        if not accepted:
            status = 'line_search_failed'
            break
    final = equation(x).detach()
    if finite(final, x) and final.norm() <= config.atol + config.rtol * torch.maximum(x.norm(), previous.norm()):
        status = 'converged'
    return dict(status=status, trace=trace, final=x, residual=final)


def check_case(record, directory, engine, device, audit, precision):
    path = Path(record['path'])
    if digest(path) != record['sha256']:
        raise ValueError(f'Capsule checksum mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    metadata = payload['metadata']
    if record['source'] in FRESH_ARMS:
        if payload.get('precision_policy') != POLICY:
            raise ValueError('Fresh capsule policy mismatch')
        if payload.get('actor_kwargs_sha256') != precision.constructor_digest(payload['actor_kwargs']):
            raise ValueError('Fresh capsule constructor hash mismatch')
        package_hashes = precision.source_hashes()
        if payload.get('package_sha256') != package_hashes:
            raise ValueError('Fresh capsule package hashes mismatch')
        source_hashes = payload.get('source_sha256', {})
        if not source_hashes or any(digest(Path(engine.__file__).parent / name) != value
                                    for name, value in source_hashes.items()):
            raise ValueError('Fresh capsule engine hashes mismatch')
        # The caller chooses the latest eligible capsule. Verify no later saved
        # capsule in that pilot directory is an eligible full actor batch.
        for later in sorted(path.parent.glob('capsule_*.pt')):
            if later.name <= path.name:
                continue
            candidate = torch.load(later, map_location='cpu', weights_only=False)
            if (candidate.get('metadata', {}).get('role') == 'training/actor_grad'
                    and candidate.get('obs', torch.empty(0)).shape[0] == 1024):
                raise ValueError(f'Fresh bank omitted later eligible capsule: {later}')
    if metadata['role'] != 'training/actor_grad' or int(metadata['global_step']) != int(record['actual_step']):
        raise ValueError('Capsule role or actual_step mismatch')
    if payload['obs'].shape[0] != 1024 or payload['z'].shape != (1024, 2):
        raise ValueError('Required full1024, action dimension2')
    for name in ('actor', 'actor_kwargs', 'q1', 'q2', 'loss_config'):
        if name not in payload:
            raise ValueError(f'Missing {name}')
    actor = engine.deq_multistep_flow_actor(**payload['actor_kwargs']).to(device).float().eval()
    actor.load_state_dict(payload['actor'], strict=True)
    actor.density_diag_every = 1
    actor._audit_precision_policy = POLICY
    actor._audit_precision_events = {}
    if actor.T != 4 or not actor.exact_backward or actor.integration != 'implicit':
        raise ValueError('Required T4, native dense backward, implicit integration')
    obs, z = payload['obs'].to(device).float(), payload['z'].to(device).float()
    if not finite(obs, z, *actor.parameters()):
        raise ValueError('Nonfinite input or actor')
    precision.assert_policy(POLICY)
    cfg = audit.NewtonConfig(rtol=THRESHOLDS['rtol'], atol=THRESHOLDS['atol'], max_iter=THRESHOLDS['max_iter'])
    seq, native = audit.solve(engine, actor, obs, z, 'newton', cfg)
    layouts = residual_layouts(actor, obs, z, seq)
    double_actor = copy.deepcopy(actor).cpu().double()
    double_actor._audit_reference_fp64 = True
    double_actor._audit_precision_events = {}
    obs64, z64, fixed64 = obs.cpu().double(), z.cpu().double(), seq.cpu().double()
    with torch.no_grad():
        residual64 = residual_layouts(double_actor, obs64, z64, fixed64)['flattened']
    boundary = classify_boundary(native['per_step']['abs_residual'].cpu(), native['per_step']['status'].cpu(),
                                 cpu(layouts), residual64, seq.cpu(), z.cpu())
    seq64, native64 = audit.solve(engine, double_actor, obs64, z64, 'newton', cfg)
    # Preserve expensive diagnostics even if a later independent check raises.
    torch.save(cpu(dict(source=record, obs=obs, z=z, trajectory=seq, native=native,
                        fixed_q_residual_layouts=layouts, fixed_q_fp64_residual=residual64,
                        boundary=boundary, trajectory_fp64=seq64, native_fp64=native64)),
               directory / 'forward_raw.pt')
    # Independent SciPy reference covers all rows. Invalid references are kept.
    ref_seq, ref_stats = audit.reference(double_actor, obs64, z64)
    status, status64 = native['per_step']['status'].cpu(), native64['per_step']['status'].cpu()
    failures = status != 0
    intrinsic = failures & (status64 != 0)
    strict_only = failures & (status64 == 0)
    fp64_only = ~failures & (status64 != 0)
    traces = {}
    for row, step in (failures | (status64 != 0)).nonzero().tolist():
        prev = z64[row] if step == 0 else seq64[row, step - 1]
        traces[f'{row}:{step}'] = trace_local(double_actor, obs64[row], prev, step, cfg)
    fd = finite_difference_gate(double_actor, obs64, seq64, audit)
    parity = parity_gate(engine, actor, obs, z, seq, payload, audit)
    precision.assert_policy(POLICY)
    blockers = []
    finite_case = finite(seq, *layouts.values(), residual64, seq64,
                         native['per_step']['abs_residual'], native['per_step']['residual_scale'])
    if not finite_case:
        blockers.append('nonfinite_solver_or_residual')
    if bool(boundary['unexplained_native_success'].any()):
        blockers.append('unexplained_native_vs_fresh_residual')
    if bool(strict_only.any()):
        blockers.append('strict_fp32_only_hard_failure')
    if bool(fp64_only.any()):
        blockers.append('fp64_only_hard_failure_requires_branch_review')
    if not fd['pass_all']:
        blockers.append('full_batch_fp64_jacobian_fd')
    if not parity['pass_all']:
        blockers.append('native_vs_audit_sac_parity')
    if bool(((status == 2) | (status == 3) | (status == 5)).any()):
        blockers.append('nonfinite_singular_or_non_descent_status_requires_review')
    ref_valid = ref_stats['valid']
    reviewed_hard = torch.zeros_like(intrinsic)
    if record['sha256'] == 'ad92d3d966bae6be75bb1c26db087f4ccfd7b67e7a75ed73184e36ef6a7937ab':
        reviewed_hard[473, 1] = True
    if bool((intrinsic & ~reviewed_hard).any()):
        blockers.append('unreviewed_fp64_reproduced_hard_failure')
    # Known adversarial fixture must stay in the full bank and keep its failure.
    if record['sha256'] == 'ad92d3d966bae6be75bb1c26db087f4ccfd7b67e7a75ed73184e36ef6a7937ab':
        if int(status[473, 1]) != 4 or int(status64[473, 1]) != 4 or bool(ref_valid[473]):
            blockers.append('historical_hard_row_changed_requires_review')
    result = dict(source=record, pass_all=not blockers, blockers=blockers,
                  native_failed_rows=int(failures.any(1).sum()),
                  fresh_failed_rows={key: int((~mask).any(1).sum()) for key, mask in boundary['exact_pass_masks'].items()},
                  boundary_disagreements=int(boundary['disagreement'].sum()),
                  explained_boundary_steps=int(boundary['explained_boundary'].sum()),
                  unexplained_boundary_steps=int(boundary['unexplained_native_success'].sum()),
                  fp64_reproduced_failed_steps=intrinsic.nonzero().tolist(),
                  unreviewed_shared_failed_steps=(intrinsic & ~reviewed_hard).nonzero().tolist(),
                  strict_only_failed_steps=strict_only.nonzero().tolist(),
                  fp64_only_failed_steps=fp64_only.nonzero().tolist(),
                  reference_valid_rows=int(ref_valid.sum()), reference_invalid_rows=int((~ref_valid).sum()),
                  full_batch_fp64_gradient_reference_available=bool(ref_valid.all()),
                  full_batch_fp64_gradient_reference_computed=False,
                  fd_pass=fd['pass_all'], parity_pass=parity['pass_all'],
                  gradient_relative=parity['gradient_relative'], precision=precision.snapshot(),
                  runtime_boundaries=actor._audit_precision_events,
                  reference_runtime_boundaries=double_actor._audit_precision_events)
    raw = dict(source=record, obs=obs, z=z, trajectory=seq, native=native,
               fixed_q_residual_layouts=layouts, fixed_q_fp64_residual=residual64,
               boundary=boundary, trajectory_fp64=seq64, native_fp64=native64,
               reference_trajectory=ref_seq, reference=ref_stats, hard_row_traces=traces,
               full_batch_fd=fd, readout_parity=parity)
    torch.save(cpu(raw), directory / 'raw.pt')
    write_json(directory / 'case.json', result)
    return result


def read_bank(path, fresh=False):
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError('Bank must be a nonempty JSON array')
    for row in records:
        if not {'path', 'sha256', 'actual_step', 'source', 'rule'} <= row.keys():
            raise ValueError('Each bank record requires path, sha256, actual_step, source, rule')
    if len({row['sha256'] for row in records}) != len(records):
        raise ValueError('Duplicate capsules in bank')
    if fresh and (len(records) != 4 or {row['source'] for row in records} != FRESH_ARMS):
        raise ValueError('Fresh bank requires one capsule from every N/A/H/T pilot')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--fixed-bank', type=Path)
    parser.add_argument('--fresh-bank', type=Path)
    parser.add_argument('--phase', choices=['fixed', 'fresh', 'all'], default='fixed')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'package'))
    from deflow_rebuttal import audit, precision
    engine_path = args.engine or root / 'DEFlow/src/docking_full/sac_anderson_flow.py'
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.set_num_threads(1)
    verdict = dict(pass_=False, phase=args.phase, thresholds=THRESHOLDS, cases=[], blockers=[],
                   root=str(root), device=args.device, torch=torch.__version__, cuda=torch.version.cuda,
                   policy=POLICY, cpu_threads=torch.get_num_threads(), script_sha256=digest(__file__), engine_sha256=digest(engine_path),
                   package_sha256={p.name: digest(p) for p in (root / 'package/deflow_rebuttal').glob('*.py')},
                   environment={k: os.environ.get(k) for k in ('NVIDIA_TF32_OVERRIDE', 'CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG')})
    try:
        precision.apply_policy(POLICY)
        precision.assert_policy(POLICY)
        verdict['precision_before'] = precision.snapshot()
        verdict['device_name'] = torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device
        engine = audit.load_engine(engine_path)
        records = []
        if args.phase in ('fixed', 'all'):
            bank = args.fixed_bank or root / 'validation/historical_fixed_bank.json'
            fixed = read_bank(bank)
            canonical = read_bank(root / 'validation/historical_fixed_bank.json')
            if len(fixed) != 3 or {r['sha256'] for r in fixed} != {r['sha256'] for r in canonical}:
                raise ValueError('Historical bank must have the three preregistered capsule hashes')
            records += fixed
            verdict['fixed_bank_sha256'] = digest(bank)
        if args.phase in ('fresh', 'all'):
            if args.fresh_bank is None:
                raise ValueError('--fresh-bank is required for fresh/all phase')
            records += read_bank(args.fresh_bank, fresh=True)
            verdict['fresh_bank_sha256'] = digest(args.fresh_bank)
        for index, record in enumerate(records):
            directory = args.output / f'case_{index:02d}'
            directory.mkdir()
            print(json.dumps({'event': 'case_start', 'index': index, 'source': record['source']}), flush=True)
            case_started = time.time()
            try:
                case = check_case(record, directory, engine, args.device, audit, precision)
            except Exception:
                case = dict(source=record, pass_all=False, blockers=['case_exception'], traceback=traceback.format_exc())
                write_json(directory / 'case.json', case)
            case['elapsed_seconds'] = time.time() - case_started
            verdict['cases'].append(case)
            verdict['blockers'] += [f"{record['source']}: {reason}" for reason in case['blockers']]
            print(json.dumps({'event': 'case_finished', 'index': index, 'pass': case['pass_all'], 'blockers': case['blockers']}), flush=True)
        precision.assert_policy(POLICY)
        verdict['precision_after'] = precision.snapshot()
    except Exception:
        verdict['blockers'].append('preflight_exception')
        verdict['traceback'] = traceback.format_exc()
    verdict['pass'] = bool(verdict['cases']) and not verdict['blockers']
    del verdict['pass_']
    verdict['elapsed_seconds'] = time.time() - started
    write_json(args.output / 'verdict.json', verdict)
    print(json.dumps({'pass': verdict['pass'], 'output': str(args.output), 'blockers': verdict['blockers']}), flush=True)
    return 0 if verdict['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
