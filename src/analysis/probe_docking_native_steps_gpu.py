"""GPU evaluation of each frozen policy with its own integrator; no training."""

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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--steps', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()
    root = args.root
    assert 0 <= args.shard < args.shards
    sys.path.insert(0, str(root / 'runtime'))
    import sac_anderson_flow as engine

    inputs = json.loads((root / 'inputs.json').read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', 'default')
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    targets = torch.tensor([[.6, .2], [-.6, -.2], [-.2, .6], [.2, -.6]], device='cuda')
    records = []
    selected = inputs['donors'][args.shard::args.shards]
    output = root / 'outputs' / f'shard{args.shard}.json'
    output.parent.mkdir(exist_ok=True)
    assert not output.exists()
    for donor in selected:
        checkpoint = root / donor['checkpoint']
        assert sha(checkpoint) == donor['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = saved['args']
        assert cfg['seed'] == donor['seed'] and cfg['denoising_steps'] == donor['T']
        assert saved['global_step'] == donor['checkpoint_step']
        integration = 'implicit' if donor['method'] == 'deflow' else 'explicit'
        assert cfg['integration'] == integration
        bank = inputs['banks'][donor['geometry']]
        latents = torch.tensor(inputs['latents'], device='cuda')
        obs = torch.tensor(bank['states'], device='cuda').repeat_interleave(len(latents), dim=0)
        goals = torch.tensor(bank['goals'], device='cuda').repeat_interleave(len(latents))
        z = latents.repeat(len(bank['states']), 1)
        for T in args.steps:
            actor = engine.deq_multistep_flow_actor(5, 2, -np.ones(2, np.float32), np.ones(2, np.float32),
                denoising_steps=T, hidden_dim=256, u_scale=cfg['u_scale'], gate_bias=cfg['gate_bias'],
                integration=integration, max_iter_fwd=25, tol_fwd=1e-5, exact_backward=True).eval()
            if T == cfg['denoising_steps']:
                actor.load_state_dict(saved['actor'], strict=True)
            else:
                actor.vf.load_state_dict({k[3:]: v for k, v in saved['actor'].items() if k.startswith('vf.')}, strict=True)
            actor.to('cuda')
            solver = None
            if integration == 'explicit':
                _, end = actor._rollout_explicit(obs, z)
            else:
                end = actor._solve_zstar_with_implicit_grad(obs, actor._make_a0_from_z(z)).reshape(len(z), T, 2)[:, -1]
                solver = {k: actor.last_fwd[k] for k in ('n_iter', 'res_last', 'res_fail_frac')}
            actions = torch.tanh(end) * actor.action_scale + actor.action_bias
            assert torch.isfinite(actions).all()
            distance = (actions - targets[goals]).norm(dim=1)
            by_goal = [float((distance[goals == g] <= 2 * bank['sigma']).float().mean()) for g in range(4)]
            records.append(dict(run_id=donor['run_id'], geometry=donor['geometry'], method=donor['method'],
                train_T=donor['T'], test_T=T, seed=donor['seed'], integration=integration,
                checkpoint_sha256=donor['checkpoint_sha256'], actions=actions.cpu().tolist(),
                distance=distance.cpu().tolist(), per_goal=by_goal, balanced_hit=float(np.mean(by_goal)), solver=solver))
        print(json.dumps(dict(run_id=donor['run_id'], completed_arms=len(records))), flush=True)
    result = dict(shard=args.shard, physical_gpu=gpu, torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  input_sha256=sha(root / 'inputs.json'), source_sha256=sha(Path(__file__)),
                  protocol='Own integration scheme only; fixed weights, common states/latents, native implicit cap25; no training', rows=records)
    with output.open('x') as handle:
        json.dump(result, handle, allow_nan=False)


if __name__ == '__main__':
    main()
