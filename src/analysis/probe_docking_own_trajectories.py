"""Replay docking on each frozen policy's own trajectories and base samples."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['explicit', 'deflow'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--donor', required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()
    root = args.root
    sys.path.insert(0, str(root / 'runtime'))
    import sac_anderson_flow as engine
    import multigoal_docking_env

    inputs = json.loads((root / 'inputs.json').read_text())
    donor = next(d for d in inputs['donors'] if d['run_id'] == args.donor and d['method'] == args.method)
    checkpoint = root / donor['checkpoint']
    assert sha(checkpoint) == donor['checkpoint_sha256']
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'default')
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['args']
    train_T = cfg['denoising_steps']
    assert train_T in args.steps
    assert saved['global_step'] == donor['checkpoint_step']
    integration = cfg['integration']
    assert integration == ('explicit' if args.method == 'explicit' else 'implicit')

    def build(T):
        actor = engine.deq_multistep_flow_actor(
            5, 2, -np.ones(2, np.float32), np.ones(2, np.float32),
            denoising_steps=T, hidden_dim=256, u_scale=cfg['u_scale'],
            gate_bias=cfg['gate_bias'], integration=integration,
            max_iter_fwd=25, tol_fwd=1e-5, exact_backward=True).eval()
        if T == train_T:
            actor.load_state_dict(saved['actor'], strict=True)
        else:
            actor.vf.load_state_dict({k[3:]: v for k, v in saved['actor'].items()
                                      if k.startswith('vf.')}, strict=True)
        return actor.to('cuda')

    actor = build(train_T)
    sample_base = actor._sample_base_z
    captured = {}

    def capture_base(B, device, dtype):
        z = sample_base(B, device, dtype)
        captured['z'] = z.detach().clone()
        return z

    actor._sample_base_z = capture_base
    n = inputs['n_episodes']
    if 'evaluation_seed' in inputs:
        raise ValueError('Evaluation randomness is configured internally')
    eval_seed = 1
    torch.manual_seed(eval_seed)
    envs = [engine.gym.make(cfg['env_id']) for _ in range(n)]
    radius = 2 * envs[0].unwrapped.docking_sigma
    obs = np.stack([e.reset(seed=eval_seed + i, options={'fixed_origin': True})[0]
                    for i, e in enumerate(envs)])
    active = np.ones(n, dtype=bool)
    returns = np.zeros(n, dtype=np.float64)
    episodes = [None] * n
    batches = []
    step = 0
    while active.any():
        step += 1
        indexes = np.flatnonzero(active)
        for start in range(0, len(indexes), 128):
            ids = indexes[start:start + 128]
            tensor = torch.as_tensor(obs[ids], device='cuda')
            docking = tensor[:, 2].bool()
            actions = actor.act(tensor, deterministic=False)
            assert torch.isfinite(actions).all()
            if docking.any():

                batches.append(dict(episode_ids=ids.tolist(), obs=tensor.cpu().tolist(),
                                    z=captured['z'].cpu().tolist(),
                                    docking_mask=docking.cpu().tolist(),
                                    actions=actions.cpu().tolist()))
            for i, action in zip(ids, actions.cpu().numpy()):
                next_obs, reward, terminated, truncated, info = envs[i].step(action)
                returns[i] += reward
                obs[i] = next_obs
                if terminated or truncated:
                    active[i] = False
                    episodes[i] = dict(episode=int(i), goal=int(info['selected_goal']),
                                       success=int(info['success']), return_value=float(returns[i]),
                                       docking_distance=info.get('docking_distance'))
        print(json.dumps(dict(stage='rollout', navigation_clock=step,
                              finished=int((~active).sum()), total=n)), flush=True)
        assert step <= 31
    for env in envs:
        env.close()

    targets = torch.tensor([[.6, .2], [-.6, -.2], [-.2, .6], [.2, -.6]], device='cuda')
    rows = []
    replay_max_error = 0.0
    for T in args.steps:
        probe = build(T)
        points, goals, ids = [], [], []
        for batch in batches:
            states = torch.tensor(batch['obs'], device='cuda')
            z = torch.tensor(batch['z'], device='cuda')
            mask = torch.tensor(batch['docking_mask'], device='cuda')
            if integration == 'explicit':
                _, end = probe._rollout_explicit(states, z)
            else:
                path = probe._solve_zstar_with_implicit_grad(states, probe._make_a0_from_z(z))
                end = path.reshape(len(z), T, 2)[:, -1]
            actions = torch.tanh(end) * probe.action_scale + probe.action_bias
            assert torch.isfinite(actions).all()
            if T == train_T:
                original = torch.tensor(batch['actions'], device='cuda')
                replay_max_error = max(replay_max_error, float((actions - original).abs().max()))
                assert torch.equal(actions, original), 'Native replay must be bit-identical'
            dock_ids = np.array(batch['episode_ids'])[np.array(batch['docking_mask'])].tolist()
            ids.extend(dock_ids)
            goals.extend(episodes[i]['goal'] for i in dock_ids)
            points.append(actions[mask])
        actions = torch.cat(points)
        assert len(set(ids)) == len(ids)
        distance = (actions - targets[goals]).norm(dim=1)
        hit = (distance <= radius).cpu().numpy()
        if T == train_T:
            assert hit.tolist() == [bool(episodes[i]['success']) for i in ids]
        counts = np.bincount(goals, minlength=4)
        by_goal = [float(hit[np.array(goals) == g].mean()) if counts[g] else None for g in range(4)]
        rows.append(dict(test_T=T, episode_ids=ids, goals=goals, actions=actions.cpu().tolist(),
                         distance=distance.cpu().tolist(), n_docking=len(ids), goal_counts=counts.tolist(),
                         docking_hit=float(hit.mean()), frozen_navigation_success=float(hit.sum() / n),
                         per_goal=by_goal))
        print(json.dumps(dict(stage='replay', T=T, docking_hit=float(hit.mean()))), flush=True)
    result = dict(method=args.method, integration=integration, donor=donor,
                  environment=cfg['env_id'], train_T=train_T, training_seed=cfg['seed'], checkpoint_step=saved['global_step'],
                  n_episodes=n, evaluation_seed=eval_seed, physical_gpu=gpu,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  source_sha256=sha(Path(__file__)), input_sha256=sha(root / 'inputs.json'),
                  runtime_sha256={p.name: sha(p) for p in (root / 'runtime').glob('*.py')},
                  protocol='Own stochastic trajectories; keep original batches and base samples; replay docking only; unweighted episode frequencies; no retraining.',
                  baseline_success=float(np.mean([e['success'] for e in episodes])),
                  baseline_mean_return=float(returns.mean()), replay_max_abs_error=replay_max_error,
                  replay_bit_identical=True, episodes=episodes, batches=batches, rows=rows)
    output = root / 'outputs' / f'{args.method}.json'
    output.parent.mkdir(exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, allow_nan=False)


if __name__ == '__main__':
    main()
