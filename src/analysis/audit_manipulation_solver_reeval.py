"""Fixed-bank trained-policy accuracy audit across the complete reevaluated cohort."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import manipulation_numerical_verification as audit


def gradient(actor, payload, obs, noise, reference=None):
    q1, q2 = audit.critics(payload, obs.dtype)
    alpha = payload['log_alpha'].exp().to(dtype=obs.dtype)
    if reference is None:
        actor._sample_base_z = lambda batch, device, dtype: noise.to(device=device, dtype=dtype)
        action, logp = actor.forward_with_logprob(obs)
        penalty = actor._jac_pen
    else:
        matrix = audit.joint_matrix(actor, audit.jacobians(actor, obs, reference)).detach()
        seq = actor._g(reference.flatten(1).detach(), actor._make_a0_from_z(noise), obs)
        seq.register_hook(lambda upstream: torch.linalg.solve(matrix.transpose(1, 2),
            upstream[..., None]).squeeze(-1))
        seq = seq.reshape_as(reference)
        action = seq[:, -1].tanh() * actor.action_scale + actor.action_bias
        components = audit.density(actor, obs, noise, seq)
        logp = components['guarded']
        norms = torch.linalg.matrix_norm(actor.dt * components['j'], ord=2)
        penalty = torch.relu(norms - actor.jac_sigma_target).square().sum(-1)
    settings = payload['args']

    cap = settings.get('logp_clip', 0.)
    if cap > 0:
        logp = logp.clamp(-cap, cap)
    inputs = torch.cat([obs, action], dim=-1)
    loss = (alpha * logp - torch.minimum(q1(inputs), q2(inputs)).squeeze(-1)).mean()
    loss = loss + float(settings['lam_jac']) * penalty.mean()
    return torch.cat([g.detach().flatten().double() for g in
                      torch.autograd.grad(loss, tuple(actor.parameters()))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.set_defaults(seed=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root
    out = root / 'numerical' / args.key
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    job = next(j for j in json.loads((root / 'bundle/manifest.json').read_text())['jobs'] if j['key'] == args.key)
    inputs = root / 'bundle/inputs' / args.key
    engine = audit.load_module(inputs / 'engine.py', 'full_cohort_audit_engine')
    assert audit.digest(inputs / 'engine.py') == job['evaluation_engine_sha256']
    assert audit.digest(inputs / 'checkpoint.pt') == job['checkpoint_sha256']
    payload = torch.load(inputs / 'checkpoint.pt', map_location='cpu', weights_only=False)
    weights = payload['actor']
    adim = weights['action_scale'].numel()
    odim = weights['vf.pre.weight'].shape[1] - adim - 32
    actors = []
    for dtype in (torch.float32, torch.float64):
        a = engine.deq_multistep_flow_actor(odim, adim,
            weights['action_bias']-weights['action_scale'], weights['action_bias']+weights['action_scale'],
            **job['original_constructor'])
        a.load_state_dict(weights)
        actors.append(a.to(dtype=dtype).eval())
    actor, actor64 = actors
    aliases = json.loads((root/'artifact_paths.json').read_text())
    folder = aliases.get(f'{args.key}_original_cpu', f'{args.key}_original_cpu')
    bank_path = root / 'remote' / folder / 'data/probes.npz'
    bank = np.load(bank_path)
    episodes = json.loads((bank_path.parent / 'result.json').read_text())['episodes']
    stochastic_steps = sum(e['steps'] for e in episodes if e['channel'] == 'stoch')
    chosen = np.linspace(0, stochastic_steps - 1, 32, dtype=int)
    obs = torch.from_numpy(np.repeat(bank['observations'][chosen], 4, axis=0))
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(128, adim, generator=generator) * actor.z_std
    sequences, solver = {}, {}
    for label, cap, tolerance in [('original', 25, 1e-5), ('tight100', 100, 1e-6)]:
        actor.tol_fwd = tolerance
        values, receipts = [], []

        for i in range(128):
            seq, _, residual, telemetry, _ = audit.solve(engine, actor, obs[i:i+1], noise[i:i+1], cap)
            values.append(seq[0]); receipts.append(telemetry)
        sequences[label] = torch.stack(values)
        solver[label] = receipts
    reference, root_residuals = [], []
    for i in range(128):
        seq, residual = audit.reference_root(actor64, obs[i].double(), noise[i].double(),
                                              sequences['original'][i].double())
        reference.append(seq); root_residuals.append(residual)
    reference = torch.stack(reference)
    root_ok = np.max(root_residuals, axis=1) < 1e-8
    reference_readout = audit.density(actor64, obs.double(), noise.double(), reference)
    action_ref = reference[:, -1].tanh() * actor64.action_scale + actor64.action_bias
    rows = []
    for label, sequence in sequences.items():
        readout = audit.density(actor, obs, noise, sequence)
        action = sequence[:, -1].tanh() * actor.action_scale + actor.action_bias
        for i in range(128):
            rows.append(dict(arm=label, sample=i, reference_ok=bool(root_ok[i]),

                action_linf=float((action[i].double()-action_ref[i]).abs().max()),
                logp_abs_nats=float((readout['guarded'][i].double()-reference_readout['guarded'][i]).abs()),
                sigma_min=float(readout['sigma_min'][i].min()),
                condition_max=float(readout['condition'][i].max()),
                determinant_clip_count=int(readout['det_clip'][i].sum()),
                squash_floor_count=int(readout['squash_floor'][i].sum())))
    gradients = []
    for start in range(0, 128, 8):
        stop = start+8
        if not root_ok[start:stop].all():
            gradients.append(dict(start=start, status='reference_failed'))
            continue
        o, z = obs[start:stop], noise[start:stop]
        reference_grad = gradient(actor64, payload, o.double(), z.double(), reference[start:stop])
        for label, cap, tolerance in [('original',25,1e-5), ('tight100',100,1e-6)]:
            actor.max_iter_fwd, actor.tol_fwd = cap, tolerance
            try:
                value = gradient(actor, payload, o, z)
                gradients.append(dict(start=start, arm=label, status='ok',
                    relative_error=float((value-reference_grad).norm()/reference_grad.norm()),
                    cosine=float(torch.nn.functional.cosine_similarity(value, reference_grad, dim=0)),
                    reference_norm=float(reference_grad.norm())))
            except RuntimeError as error:
                gradients.append(dict(start=start, arm=label, status='failed', error=str(error)))
        print(json.dumps({'key':args.key, 'gradient_inputs':stop}), flush=True)
    np.savez_compressed(out/'inputs.npz', obs=obs.numpy(), noise=noise.numpy(), reference=reference.numpy(),
                        **{k:v.numpy() for k,v in sequences.items()})
    result = dict(job=job, rows=rows, gradients=gradients, root_residuals=root_residuals,
        root_failures=int((~root_ok).sum()), solver=solver,
        source_bank_sha256=audit.digest(bank_path), script_sha256=audit.digest(__file__),
        helper_sha256=audit.digest(audit.__file__), runtime=audit.runtime(),
        wall_seconds=time.perf_counter()-started, actor_logp_rail=payload['args'].get('logp_clip',0.),
        selection='32 evenly spaced stochastic-trajectory states with four Gaussian inputs each.',
        scope='CPU FP32/FP64 arithmetic; B1 action/density and B8 gradient tests before clipping; saved PushWall actor logp rail retained. No finite-sampler likelihood certification.')
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
