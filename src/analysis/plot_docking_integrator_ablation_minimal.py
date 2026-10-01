"""Minimal publication layouts for the shared-state and own-trajectory T sweeps."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-inputs', type=Path, required=True)
    parser.add_argument('--shared-results', type=Path, nargs='+', required=True)
    parser.add_argument('--own-results', type=Path, nargs='+', required=True)
    parser.add_argument('--bank', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()
    OUT = args.output_dir
    sources = {}

    def read(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    inputs = read(args.shared_inputs)
    common_goals = np.repeat(inputs['banks'][args.bank]['goals'], len(inputs['latents']))
    radius = 2 * inputs['banks'][args.bank]['sigma']
    common_rows = [r for p in args.shared_results
                   for r in read(p)['rows']]
    own = {}
    for path in args.own_results:
        payload = read(path)
        assert payload['method'] not in own
        own[payload['method']] = payload
    targets = np.array([[.6, .2], [-.6, -.2], [-.2, .6], [.2, -.6]], dtype=np.float32)
    colors = ['#3479BB', '#D88333', '#53A26B', '#AD6196']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []
    for protocol in ('shared_states', 'own_trajectories'):
        fig, axes = plt.subplots(2, len(args.steps), squeeze=False, figsize=(10.4, 5.2), sharex=True, sharey=True,
                                 constrained_layout=True)
        for i, (method, label) in enumerate([('explicit', 'Explicit'), ('deflow', 'Implicit')]):
            run_id = own[method]['donor']['run_id']
            for j, T in enumerate(args.steps):
                if protocol == 'shared_states':
                    row = next(r for r in common_rows if r['run_id'] == run_id and r['test_T'] == T and r['geometry'] == args.bank)
                    goals = common_goals
                    expected = row['balanced_hit']
                else:
                    assert own[method]['donor']['run_id'] == run_id
                    assert own[method]['replay_bit_identical']
                    row = next(r for r in own[method]['rows'] if r['test_T'] == T)
                    goals = np.array(row['goals'])
                    expected = row['docking_hit']
                actions = np.array(row['actions'], dtype=np.float32)
                assert actions.shape == (len(goals), 2) and np.isfinite(actions).all()
                hits = np.linalg.norm(actions - targets[goals], axis=1) <= np.float32(radius)
                metric = np.mean([hits[goals == g].mean() for g in range(4)]) if protocol == 'shared_states' else hits.mean()
                assert abs(metric - expected) < 1e-7
                ax = axes[i, j]
                for g, (center, color) in enumerate(zip(targets, colors)):
                    points = actions[goals == g]
                    ax.scatter(points[:, 0], points[:, 1], color=color, s=5,
                               alpha=.4, linewidths=0, rasterized=False)
                    target_color = color if len(points) else '#bcbcbc'
                    ax.scatter(*center, color=target_color, marker='x', s=20, zorder=4)
                    ax.add_patch(Circle(center, radius, fill=False, color=target_color, lw=.9))
                ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal',
                       xticks=[-1, 0, 1], yticks=[-1, 0, 1])
                ax.grid(alpha=.12)
                if i == 0:
                    ax.set_title(f'T = {T}', fontsize=12, pad=7)
                if j == 0:
                    ax.set_ylabel(f'{label}\nAction y', fontsize=11)
                if i == 1:
                    ax.set_xlabel('Action x', fontsize=10)
                ax.text(.035, .965, f'Hit {100 * metric:.1f}%', transform=ax.transAxes,
                        va='top', fontsize=10,
                        bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=1))
                checks.append(dict(protocol=protocol, method=method, train_T=own[method]['train_T'], replicate=own[method].get('replicate', own[method].get('training_seed')),
                                   test_T=T, n_actions=len(actions), hit=float(metric),
                                   actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest()))
        for extension in ('png', 'pdf'):
            fig.savefig(OUT / f'{protocol}_minimal.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    result = dict(panels_verified=len(checks), sources=sources, checks=checks,
                  plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  changes='Layout only; all raw points preserved; no coordinate jitter or filtering; vector PDF scatter.')
    (OUT / 'minimal_figure_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(panels_verified=len(checks), figures=2)))


if __name__ == '__main__':
    main()
