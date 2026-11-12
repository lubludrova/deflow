"""Frozen full-batch solver audit. Trusted checkpoint/capsule files only.

Reference: independent sequential scipy hybr in FP64, validated by local
residuals and nonsingularity. This does NOT certify finite-sampler likelihood.
Native mixed-point g(q) attachment, shifted adjoint, density guards are retained
for both production solvers. Optional FD tests the implemented finite program.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
import torch
import numpy as np
from torch.func import jacrev, vmap
from .sequential_newton import NewtonConfig, solve_sequential
from .run_docking import anderson_status
from .precision import apply_policy, assert_policy, assert_float32, constructor_digest, source_hashes


def load_engine(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('audit_docking_engine', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def jacobians(actor, obs, seq):
    batch, steps, dim = seq.shape
    times = torch.arange(1, steps + 1, device=seq.device, dtype=seq.dtype) * actor.dt
    def field(a, s, t):
        return actor.vf(a[None], s[None], t[None])[0]
    return vmap(jacrev(field, argnums=0))(
        seq.reshape(-1, dim), obs[:, None].expand(-1, steps, -1).reshape(batch*steps, -1),
        times[None, :, None].expand(batch, -1, -1).reshape(-1, 1)
    ).reshape(batch, steps, dim, dim)


def local_residual(actor, obs, z, seq):
    previous = torch.cat([z[:, None], seq[:, :-1]], dim=1)
    times = (torch.arange(1, actor.T+1, device=seq.device, dtype=seq.dtype)*actor.dt)
    values = actor.vf(seq.flatten(0, 1), obs[:, None].expand(-1, actor.T, -1).flatten(0, 1),
                      times[None, :, None].expand(len(z), -1, -1).flatten(0, 1)).view_as(seq)
    residual = seq - previous - actor.dt*values
    absolute = residual.norm(dim=-1)
    relative = absolute/(1 + previous.norm(dim=-1))
    return absolute.detach(), relative.detach()


def observe_precision(actor, event, *tensors):
    policy=getattr(actor, '_audit_precision_policy', None)
    if policy is None:
        return
    actual=assert_policy(policy)
    if not getattr(actor, '_audit_reference_fp64', False):
        assert_float32(policy, *tensors, *actor.parameters())
    actor._audit_precision_events.setdefault(event, dict(actual=actual,
        dtypes=sorted({str(t.dtype) for t in tensors}),
        devices=sorted({str(t.device) for t in tensors})))


def attach(actor, obs, z, seq):
    """Exactly the historical mixed-point attachment and shifted dense adjoint."""
    q = seq.detach().flatten(1)
    r = actor._g(q, actor._make_a0_from_z(z), obs)
    j = jacobians(actor, obs, r.detach().view_as(seq)).detach()
    block = torch.stack([torch.block_diag(*sample.unbind()) for sample in j])
    eye = torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
    matrix = (eye - actor.dt*(actor.matrix_in_g.to(q) @ block)).transpose(1, 2) + 1e-6*eye
    if r.requires_grad:
        def backward(grad):
            observe_precision(actor, 'backward_entry', grad)
            solution = torch.linalg.solve(matrix, grad[..., None])[..., 0]
            residual = (matrix @ solution[..., None])[..., 0]-grad
            actor._audit_adjoint = {
                'shifted_relative_residual': (residual.norm(dim=-1)/(grad.norm(dim=-1)+1e-12)).detach(),
                'unshifted_relative_residual': ((residual-1e-6*solution).norm(dim=-1)/(grad.norm(dim=-1)+1e-12)).detach(),
                'solution_norm': solution.norm(dim=-1).detach()}
            observe_precision(actor, 'backward_result', solution)
            return solution
        r.register_hook(backward)
    return r


def reference(actor, obs, z, tolerance=1e-9):
    """Independent z/previous-root starts, never initialized from either candidate."""
    from scipy.optimize import root
    a = copy.deepcopy(actor).cpu().double()
    observations, noise = obs.cpu().double(), z.cpu().double()
    trajectories, scipy_success = [], []
    for s, initial in zip(observations, noise):
        prev = initial.clone()
        points, status = [], []
        for k in range(a.T):
            t = torch.tensor([[(k+1)*a.dt]], dtype=torch.float64)
            def equation(x):
                with torch.no_grad():
                    value = a.vf(torch.as_tensor(x)[None], s[None], t)[0]
                return (torch.as_tensor(x) - prev - a.dt*value).numpy()
            def derivative(x):
                def vf(u):
                    return a.vf(u[None], s[None], t)[0]
                return (torch.eye(a.act_dim, dtype=torch.float64)-a.dt*jacrev(vf)(torch.as_tensor(x))).detach().numpy()
            try:
                answer = root(equation, prev.numpy(), jac=derivative, method='hybr', tol=1e-11)
                prev = torch.as_tensor(answer.x).clone()
                success = bool(answer.success)
            except (RuntimeError, ValueError):
                prev = torch.full_like(prev, float('nan'))
                success = False
            points.append(prev)
            status.append(success)
        trajectories.append(torch.stack(points))
        scipy_success.append(status)
    seq = torch.stack(trajectories)
    absolute, relative = local_residual(a, observations, noise, seq)
    valid = torch.isfinite(seq).all(dim=(1, 2)) & (absolute.max(dim=1).values <= tolerance)
    sigma_min = torch.full((len(seq), a.T), float('nan'), dtype=torch.float64)
    for i in valid.nonzero().flatten().tolist():
        matrices = torch.eye(a.act_dim, dtype=torch.float64)-a.dt*jacobians(a, observations[i:i+1], seq[i:i+1])
        sigma_min[i] = torch.linalg.svdvals(matrices)[0, :, -1]
    valid &= (sigma_min > 1e-10).all(dim=1)
    return seq, {'valid': valid, 'scipy_success': torch.tensor(scipy_success),
                 'absolute_residual': absolute, 'relative_residual': relative, 'sigma_min': sigma_min}


def critics(payload, device, dtype):
    result = []
    for key in ('q1', 'q2'):
        state = payload[key]
        indices = sorted(int(k.split('.')[1]) for k in state if k.endswith('weight'))
        layers = []
        for i in indices:
            w = state[f'net.{i}.weight']
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
            if i != indices[-1]:
                layers.append(torch.nn.ReLU())
        net = torch.nn.Sequential(*layers).to(device=device, dtype=dtype)
        net.load_state_dict({k.removeprefix('net.'): v for k, v in state.items()})
        net.requires_grad_(False)
        result.append(net)
    return result


def readout(actor, obs, z, seq, qnets=None, loss_config=None):
    """Use original density and penalty code, replacing only the computed root."""
    old_sample = actor._sample_base_z
    old_solve = actor._solve_zstar_with_implicit_grad
    actor._sample_base_z = lambda *args: z
    actor._solve_zstar_with_implicit_grad = lambda s, a0: attach(actor, s, z, seq)
    try:
        actor._jac_pen = None
        observe_precision(actor, 'density_entry', obs, seq)
        action, logp = actor.forward_with_logprob(obs)
        observe_precision(actor, 'density_result', action, logp)
        if qnets is None:
            loss = (action.square().sum(-1) + .1*logp).mean()
        else:
            lp = logp.clamp(loss_config['logp_lower'], loss_config['logp_upper'])
            q = torch.minimum(*(net(torch.cat([obs, action], dim=-1)) for net in qnets)).squeeze(-1)
            loss = (loss_config['alpha']*lp-q).mean()
            if loss_config['effective_lam_jac']:
                if actor._jac_pen is None:
                    raise ValueError('Requested Jacobian penalty absent in actor')
                loss = loss + loss_config['effective_lam_jac']*actor._jac_pen.mean()
        grads = torch.autograd.grad(loss, list(actor.parameters()), allow_unused=True)
        gradient = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                              for p, g in zip(actor.parameters(), grads)])
        return {'action': action.detach(), 'logp': logp.detach(), 'loss': loss.detach(),
                'gradient': gradient.detach()}
    finally:
        actor._sample_base_z = old_sample
        actor._solve_zstar_with_implicit_grad = old_solve


def solve(engine, actor, obs, z, method, config):
    observe_precision(actor, 'solver_entry', obs, z)
    if method == 'anderson':
        with torch.no_grad():
            flat, stats = engine.anderson(lambda x: actor._g(x, actor._make_a0_from_z(z), obs),
                actor._make_a0_from_z(z).clone(), m=actor.m_fwd, lam=actor.lam_fwd,
                max_iter=actor.max_iter_fwd, tol=actor.tol_fwd, beta=actor.beta_fwd, return_stats=True)
        return flat.view(len(z), actor.T, actor.act_dim), stats
    return solve_sequential(actor.vf, obs, z, steps=actor.T, dt=actor.dt, config=config)


def stats(values):
    values = values.detach().double().flatten().cpu()
    finite = values[torch.isfinite(values)]
    return {'n': len(values), 'nonfinite': int(len(values)-len(finite)),
            'median': float(finite.median()) if len(finite) else None,
            'p95': float(torch.quantile(finite, .95)) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None}


def clean(value):
    """Serialize constructor tensors and NumPy metadata without nonstandard JSON."""
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision-policy',choices=['strict_fp32','legacy_tf32'],required=True)
    parser.add_argument('--require-same-capsule',action='store_true')
    parser.add_argument('--engine', required=True, help='Exact historical docking sac_anderson_flow.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--inputs', required=True, help='Trusted .pt capsule with obs, z, optionally actor_kwargs')
    parser.add_argument('--actor-config', help='JSON constructor kwargs, required if absent from capsule')
    parser.add_argument('--loss-config', help='JSON alpha, logp_lower, logp_upper, effective_lam_jac')
    parser.add_argument('--probe-loss', action='store_true', help='Explicitly use the labeled surrogate even if SAC metadata exists')
    parser.add_argument('--repeats', type=int, default=5, help='Synchronized solver-only timing repetitions')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', required=True, help='New directory, never overwritten')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--anderson-tol', type=float, default=None)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--anderson-cap', type=int, default=None)
    limits.add_argument('--anderson-caps', type=int, nargs='+', help='Paired fixed-input sweep, e.g. 25 100 500. Reference computed once.')
    parser.add_argument('--newton-tol', type=float, default=1e-5)
    parser.add_argument('--newton-cap', type=int, default=25)
    parser.add_argument('--fd-step', type=float, default=0., help='Optional centered finite-program parameter-direction check')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or (args.probe_loss and args.loss_config):
        parser.error('Positive repeats, nonnegative warmup, and mutually exclusive probe/loss-config required')
    if args.anderson_caps and (min(args.anderson_caps)<3 or len(set(args.anderson_caps))!=len(args.anderson_caps)):
        parser.error('Sweep caps must be unique and at least3')
    apply_policy(args.precision_policy)
    if args.require_same_capsule and digest(args.checkpoint)!=digest(args.inputs):
        raise ValueError('Campaign audit requires identical checkpoint/input capsule bytes')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    inputs = torch.load(args.inputs, map_location='cpu', weights_only=False)
    kwargs = json.loads(Path(args.actor_config).read_text()) if args.actor_config else inputs['actor_kwargs']
    if args.require_same_capsule:
        if args.actor_config or args.loss_config or args.probe_loss:
            raise ValueError('Campaign audit uses only the recorded constructor and SAC loss')
        if payload.get('precision_policy')!=args.precision_policy:
            raise ValueError('Capsule precision policy mismatch')
        if constructor_digest(kwargs)!=payload.get('actor_kwargs_sha256'):
            raise ValueError('Capsule constructor mismatch')
        recorded_sources=payload.get('source_sha256',{})
        if not recorded_sources or Path(args.engine).name not in recorded_sources:
            raise ValueError('Capsule engine source manifest missing')
        for name, expected in recorded_sources.items():
            if digest(Path(args.engine).parent/name)!=expected:
                raise ValueError(f'Capsule engine source mismatch: {name}')
        current=source_hashes()
        for name in ('sequential_newton.py','run_docking.py','precision.py','audit.py'):
            if payload.get('package_sha256',{}).get(name)!=current[name]:
                raise ValueError(f'Capsule package mismatch: {name}')
        if not payload.get('loss_config') or not all(k in payload for k in ('q1','q2')):
            raise ValueError('Campaign audit requires recorded SAC critics and loss')
    engine = load_engine(args.engine)
    actor = engine.deq_multistep_flow_actor(**kwargs).to(args.device).float()
    actor.load_state_dict(payload['actor'], strict=True)
    actor.eval()
    actor._audit_precision_policy=args.precision_policy
    actor._audit_precision_events={}
    if args.anderson_tol is not None:
        if args.anderson_tol <= 0: parser.error('anderson-tol must be positive')
        actor.tol_fwd = args.anderson_tol
    if args.anderson_cap is not None:
        if args.anderson_cap < 3: parser.error('anderson-cap must be at least3 for native implementation')
        actor.max_iter_fwd = args.anderson_cap
    actor.density_diag_every = 1
    if not actor.exact_backward or actor.integration != 'implicit':
        raise ValueError('Audit requires exact_backward=True and implicit integration')
    obs, z = inputs['obs'].to(args.device).float(), inputs['z'].to(args.device).float()
    if obs.ndim != 2 or z.shape != (len(obs), actor.act_dim) or not len(obs):
        raise ValueError('Invalid full-batch input shapes')
    if not (torch.isfinite(obs).all() and torch.isfinite(z).all()):
        raise ValueError('Input bank contains nonfinite entries')
    loss_config = (json.loads(Path(args.loss_config).read_text()) if args.loss_config
                   else payload.get('loss_config'))
    if args.probe_loss:
        loss_config = None
    if loss_config is not None:
        required = {'alpha', 'logp_lower', 'logp_upper', 'effective_lam_jac'}
        if not required.issubset(loss_config):
            raise ValueError('Incomplete SAC loss_config metadata')
        loss_config = dict(loss_config)
        loss_config['logp_lower'] = -float('inf') if loss_config['logp_lower'] is None else loss_config['logp_lower']
        loss_config['logp_upper'] = float('inf') if loss_config['logp_upper'] is None else loss_config['logp_upper']
    qnets = critics(payload, args.device, torch.float32) if loss_config else None
    cfg = NewtonConfig(rtol=args.newton_tol, max_iter=args.newton_cap)
    assert_float32(args.precision_policy,obs,z,*actor.parameters())
    summary = {'precision_policy':args.precision_policy,'runtime_precision':assert_policy(args.precision_policy),
        'package_sha256':source_hashes(),'input_kind':'campaign_same_capsule' if args.require_same_capsule else 'historical_or_cross_checkpoint_explicit',
        'manifest': vars(args), 'torch': torch.__version__, 'batch': len(obs),
        'hashes': {key: digest(getattr(args, key)) for key in ('engine', 'checkpoint', 'inputs', 'actor_config', 'loss_config') if getattr(args,key)},
        'effective_anderson': {'tol':actor.tol_fwd,'cap':actor.max_iter_fwd}, 'actor_kwargs': kwargs, 'newton_config': vars(cfg), 'audit_density_diag_every': 1,
        'reference_kind': 'converged safeguarded equilibrium readout with common shifted adjoint, not exact finite-program density',
        'device_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device,
        'loss_kind': 'configured_SAC_actor_loss' if qnets else 'probe_action_squared_plus_0.1_logp_NOT_SAC',
        'limitations': ['Reference root branch may differ from candidate: report branch differences, do not silently exclude.',
                       'Reference tests equilibrium agreement, not exact finite-sampler likelihood.',
                       'Frozen input bank does not measure full-training nonconvergence frequency.',
                       'Solver-only timings include solver diagnostics/counter extraction, exclude readout and training telemetry. Timing repeats are not independent training seeds.',
                       'Density/backward safeguards and mixed-point attachment retained for both candidates.']}
    raw = {'obs': obs.cpu(), 'z': z.cpu()}
    ref_seq, ref_stats = reference(actor, obs, z)
    raw['reference'] = {'trajectory': ref_seq, **ref_stats}
    summary['reference_valid'] = int(ref_stats['valid'].sum())
    summary['reference_invalid'] = int((~ref_stats['valid']).sum())
    # Batch gradient reference is meaningful only if every input is validated.
    ref_actor = copy.deepcopy(actor).double()
    ref_actor._audit_reference_fp64=True
    ref_actor._audit_precision_events={}
    ref_q = critics(payload, args.device, torch.float64) if loss_config else None
    ref_readout = None
    if ref_stats['valid'].all():
        try:
            ref_readout = readout(ref_actor, obs.double(), z.double(), ref_seq.to(args.device), ref_q, loss_config)
            raw['reference_readout'] = {k: v.cpu() for k,v in ref_readout.items()}
        except (RuntimeError, ValueError) as error:
            summary['reference_readout_error'] = str(error)
    caps = args.anderson_caps or [actor.max_iter_fwd]
    candidates = [('anderson', cap, f'anderson_cap{cap}' if args.anderson_caps else 'anderson') for cap in caps]
    candidates.append(('newton', actor.max_iter_fwd, 'newton'))
    summary['anderson_sweep_caps'] = caps
    for method, cap, label in candidates:
        actor.max_iter_fwd = cap
        for _ in range(args.warmup):
            solve(engine, actor, obs, z, method, cfg)
        timings = []
        for _ in range(args.repeats):
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            start = time.perf_counter()
            seq, native = solve(engine, actor, obs, z, method, cfg)
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        seconds = float(torch.tensor(timings).median())
        absolute, relative = local_residual(actor, obs, z, seq)
        record = {'forward_seconds_median': seconds, 'forward_seconds_repetitions': timings,
                  'forward_seconds_distribution': stats(torch.tensor(timings)), 'local_absolute': stats(absolute),
                  'local_relative': stats(relative), 'reference_gradient_available': ref_readout is not None}
        if method == 'anderson':
            native.update(anderson_status(native,actor.tol_fwd,cap))
            record['convergence'] = native
        raw[label] = {'trajectory': seq.detach().cpu(), 'native_stats': native,
                       'absolute_residual': absolute.cpu(), 'relative_residual': relative.cpu()}
        if torch.isfinite(seq).all():
            try:
                result = readout(actor, obs, z, seq, qnets, loss_config)
            except (RuntimeError, ValueError) as error:
                record['readout_error'] = str(error)
                summary[label] = record
                continue
            raw[label]['readout'] = {k:v.cpu() for k,v in result.items()}
            record['native_density_diagnostics'] = dict(actor.last_diag)
            adjoint = getattr(actor, '_audit_adjoint', {})
            raw[label]['adjoint'] = {k:v.cpu() for k,v in adjoint.items()}
            record['adjoint'] = {k:stats(v) for k,v in adjoint.items()}
            with torch.no_grad():
                attached = actor._g(seq.flatten(1), actor._make_a0_from_z(z), obs).view_as(seq)
                attached_absolute, attached_relative = local_residual(actor, obs, z, attached)
            raw[label]['attached_trajectory'] = attached.cpu()
            record['attached_local_relative'] = stats(attached_relative)
            record['attachment_delta'] = stats((attached-seq).flatten(1).norm(dim=1))
            record['loss'] = float(result['loss'])
            valid = ref_stats['valid'].to(args.device)
            record['trajectory_distance_to_reference_valid_rows'] = stats(
                (seq[valid].double()-ref_seq.to(args.device)[valid]).flatten(1).norm(dim=1))
            if ref_readout:
                record['action_difference'] = stats((result['action'].double()-ref_readout['action']).norm(dim=-1))
                record['logp_abs_difference'] = stats((result['logp'].double()-ref_readout['logp']).abs())
                g, rg = result['gradient'].double(), ref_readout['gradient']
                record['gradient_absolute_difference'] = float((g-rg).norm())
                record['gradient_relative_difference'] = float((g-rg).norm()/rg.norm().clamp_min(1e-12))
                record['gradient_reference_norm'] = float(rg.norm())
            if args.fd_step > 0:
                generator = torch.Generator(device=args.device).manual_seed(126152)
                direction = torch.randn(result['gradient'].shape, device=args.device, generator=generator)
                direction /= direction.norm()
                original = [p.detach().clone() for p in actor.parameters()]
                values = []
                try:
                    for sign in (-1, 1):
                        offset = 0
                        with torch.no_grad():
                            for p, old in zip(actor.parameters(), original):
                                n = p.numel()
                                p.copy_(old+sign*args.fd_step*direction[offset:offset+n].view_as(p))
                                offset += n
                        perturbed, _ = solve(engine, actor, obs, z, method, cfg)
                        values.append(float(readout(actor, obs, z, perturbed, qnets, loss_config)['loss']))
                finally:
                    with torch.no_grad():
                        for p, old in zip(actor.parameters(), original): p.copy_(old)
                record['finite_program_directional_fd'] = (values[1]-values[0])/(2*args.fd_step)
                record['implicit_directional_derivative'] = float(result['gradient']@direction)
                record['fd_caveat'] = 'One FP32 step/direction, branch changes and roundoff possible. Not a gradient certification.'
        else:
            record['readout_error'] = 'nonfinite trajectory, no rows removed to manufacture full-batch gradient'
        summary[label] = record
    if args.anderson_caps:
        high = f'anderson_cap{max(caps)}'
        summary['paired_vs_highest_cap'] = {}
        for cap in caps:
            label = f'anderson_cap{cap}'
            left, right = raw[label], raw[high]
            paired = {'trajectory_l2': stats((left['trajectory']-right['trajectory']).flatten(1).norm(dim=1))}
            if 'readout' in left and 'readout' in right:
                a, b = left['readout'], right['readout']
                paired.update(action_l2=stats((a['action']-b['action']).norm(dim=-1)),
                              logp_absolute=stats((a['logp']-b['logp']).abs()),
                              gradient_relative=float((a['gradient']-b['gradient']).norm()/b['gradient'].norm().clamp_min(1e-12)),
                              high_cap_gradient_norm=float(b['gradient'].norm()))
            summary['paired_vs_highest_cap'][label] = paired
        summary['paired_caveat'] = 'Highest cap is not a convergence certificate. Check native mean criterion and per-sample/local residual tails.'
    assert_policy(args.precision_policy)
    summary['runtime_boundaries']=actor._audit_precision_events
    summary['reference_fp64_runtime_boundaries']=ref_actor._audit_precision_events
    torch.save(raw, output/'raw.pt')
    # Preserve all nonfinite observations in raw.pt, use strict JSON nulls in summary.
    (output/'summary.json').write_text(json.dumps(clean(summary), indent=2, allow_nan=False)+'\n')
    print(output/'summary.json')


if __name__ == '__main__':
    main()
