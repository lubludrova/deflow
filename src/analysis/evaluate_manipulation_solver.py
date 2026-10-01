"""Paired frozen-checkpoint evaluation; preserves the archived episode function."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--arm', choices=['original', 'tight100'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    identity = {'pid': os.getpid(), 
                'hostname': platform.node(), 'command': sys.argv, 'runner_sha256': sha(__file__),
                'manifest_sha256': sha(args.bundle / 'manifest.json')}
    atomic(args.out / 'launch.json', identity)
    try:
        evaluate(args, identity, started)
    except BaseException as error:
        atomic(args.out / 'terminal.json', {'exit_code': 1, 'identity': identity,
            'wall_seconds': time.perf_counter() - started, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def evaluate(args, identity, started):
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['key'] == args.key)
    inputs = args.bundle / 'inputs' / args.key
    assert sha(inputs / 'checkpoint.pt') == job['checkpoint_sha256']
    assert sha(inputs / 'contract.json') == job['contract_sha256']
    assert sha(inputs / 'engine.py') == job['evaluation_engine_sha256']
    for filename, expected in manifest['source_hashes'].items():
        assert sha(args.bundle / filename) == expected, filename
    if job['task'].startswith('ms_'):
        import sapien
        print(sapien.render.get_device_summary(), flush=True)
    import numpy as np
    import torch
    from importlib.metadata import version
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if args.device == 'cuda':
        assert torch.cuda.is_available()
    source = args.bundle / 'source'
    sys.path[:0] = [str(source), str(source / 'repro'), str(source / 'analysis')]
    engine = load(inputs / 'engine.py', 'sac_anderson_flow')
    actor_class = engine.deq_multistep_flow_actor
    evaluator = load(source / 'analysis/evaluate_stage1_panel.py', 'frozen_evaluator')
    assert engine.deq_multistep_flow_actor is actor_class, 'Actor import-order substitution'
    env_alias = 'mw_peg' if job['task'] == 'mw_peg_native' else job['task']
    if job['task'] == 'mw_peg_native':
        evaluator.stage1.STAGE3_ENV_RULES['mw_peg'] = dict(divisor=1., success_bonus=100.,
            success_reward=0., horizon=200, terminal_bonus=0., terminate_on_success=False)
    payload = torch.load(inputs / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert payload['global_step'] == job['checkpoint_step']
    weights = payload['actor']
    act_dim = weights['action_scale'].numel()
    obs_dim = weights['vf.pre.weight'].shape[1] - act_dim - 32
    scale, bias = weights['action_scale'], weights['action_bias']
    config = dict(job['original_constructor'])
    config.update(manifest['arms'][args.arm])
    actor = actor_class(obs_dim, act_dim, bias - scale, bias + scale, **config).to(args.device).eval()
    actor.load_state_dict(weights, strict=True)
    low, high = (bias - scale).numpy(), (bias + scale).numpy()
    runtime = dict(python=sys.version, platform=platform.platform(), device=args.device,
        torch_cuda=torch.version.cuda, packages={p: version(p) for p in
            ['torch', 'numpy', 'scipy', 'gymnasium', 'mani-skill', 'sapien', 'metaworld', 'mujoco']},
        precision=torch.get_float32_matmul_precision(), threads=torch.get_num_threads())
    telemetry, obs_bank, noise_bank, trajectory_bank = [], [], [], []
    native_anderson = engine.anderson

    def monitored(f, x0, *pos, **kw):
        requested = kw.pop('return_stats', False)
        solved, stats = native_anderson(f, x0, *pos, **kw, return_stats=True)
        telemetry.append([stats['n_iter'], stats['res_last'], stats['abs_res_max']])
        trajectory_bank.append(solved.detach().cpu().numpy().reshape(actor.T, act_dim))
        return (solved, stats) if requested else solved


    probe = torch.zeros(1, obs_dim, device=args.device)
    parity = []
    for deterministic in (False, True):
        torch.manual_seed(manifest['policy_seeds'][0])
        with torch.no_grad():
            original = actor.act(probe, deterministic)
        engine.anderson = monitored
        torch.manual_seed(manifest['policy_seeds'][0])
        with torch.no_grad():
            instrumented = actor.act(probe, deterministic)
        delta = float((original - instrumented).abs().max())


        cached_root = torch.as_tensor(trajectory_bank[-1], device=args.device).reshape(1, -1)
        def replay_root(f, x0, *pos, **kw):
            return (cached_root.clone(), {}) if kw.get('return_stats') else cached_root.clone()
        engine.anderson = replay_root
        torch.manual_seed(manifest['policy_seeds'][0])
        with torch.no_grad():
            replayed = actor.act(probe, deterministic)
        assert torch.equal(replayed, instrumented), 'Telemetry changed the returned action'
        parity.append(delta)
        engine.anderson = native_anderson
    engine.anderson = monitored
    native_act = actor.act
    call_times = []

    def observed_act(obs, deterministic=False):
        obs_bank.append(obs.detach().cpu().numpy().reshape(-1))

        sampled = actor._sample_base_z
        def record_noise(*a, **kw):
            z = sampled(*a, **kw)
            noise_bank.append(z.detach().cpu().numpy().reshape(-1))
            return z
        actor._sample_base_z = record_noise
        if deterministic:
            noise_bank.append(np.zeros(act_dim, dtype=np.float32))
        if args.device == 'cuda':
            torch.cuda.synchronize()
        tick = time.perf_counter()
        try:
            action = native_act(obs, deterministic)
        finally:
            actor._sample_base_z = sampled
        if args.device == 'cuda':
            torch.cuda.synchronize()
        call_times.append(time.perf_counter() - tick)
        if not torch.isfinite(action).all():
            raise RuntimeError('Nonfinite policy action')
        return action

    actor.act = observed_act
    telemetry.clear()
    trajectory_bank.clear()
    episodes = []
    env_seeds = manifest['env_seeds'][:1] if args.pilot else manifest['env_seeds']
    for channel in ['stoch', 'det']:
        policy_seeds = (manifest['policy_seeds'][:1] if args.pilot or channel == 'det' else manifest['policy_seeds'])
        for env_seed in env_seeds:
            for policy_seed in policy_seeds:
                if time.perf_counter() - started > manifest['max_job_seconds']:
                    raise TimeoutError('Fixed job wall cap exceeded')
                env = evaluator.make_eval_env(env_alias, env_seed)
                try:
                    ep = evaluator.run_episode(actor, engine, env, env_seed, policy_seed,
                        channel, job['horizon'], torch.device(args.device), low, high)
                finally:
                    env.close()
                assert 0 < ep['steps'] <= job['horizon'], ep
                episodes.append(ep)
                with (args.out / 'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(ep) + '\n')
                print(json.dumps({'episodes': len(episodes), 'channel': channel,
                    'wall_seconds': time.perf_counter() - started, 'success': ep['native_ever_success']}), flush=True)
    aggregate = {}
    for channel in ['stoch', 'det']:
        selected = [e for e in episodes if e['channel'] == channel]
        aggregate[channel] = {k: float(np.mean([e[k] for e in selected])) for k in
            ['native_ever_success', 'success_at_horizon', 'native_dense_return']}
    arrays = dict(observations=np.array(obs_bank), noise=np.array(noise_bank),
        trajectories=np.array(trajectory_bank), solver=np.array(telemetry),
        action_seconds=np.array(call_times))
    assert all(len(a) == sum(e['steps'] for e in episodes) for a in arrays.values())
    np.savez_compressed(args.out / 'probes.npz', **arrays)
    result = dict(job=job, arm=args.arm, constructor=config, episodes=episodes,
        aggregate=aggregate, runtime=runtime, instrumented_api_max_abs_difference=parity,
        fixed_root_instrumentation_parity='bitwise equal; fresh-root CPU repeat variability recorded separately',
        inference_seconds=sum(call_times), wall_seconds=time.perf_counter() - started,
        solver_cap_fraction=float(np.mean(arrays['solver'][:, 0] == actor.max_iter_fwd)),
        solver_tolerance_fail_fraction=float(np.mean(arrays['solver'][:, 1] >= actor.tol_fwd)),
        solver_relative_residual_quantiles=np.quantile(arrays['solver'][:, 1], [.5, .95, .99, 1]).tolist(),
        inference_ms_quantiles=(1000*np.quantile(call_times, [.5, .95, .99])).tolist(),
        probes_sha256=sha(args.out / 'probes.npz'), pilot=args.pilot,
        timing_note='Includes identical telemetry, CUDA synchronization and noise capture; not bare actor throughput.')
    atomic(args.out / 'result.json', result)
    atomic(args.out / 'terminal.json', dict(exit_code=0, identity=identity, episodes=len(episodes),
        steps=len(call_times), result_sha256=sha(args.out / 'result.json'), wall_seconds=time.perf_counter() - started))


if __name__ == '__main__':
    main()
