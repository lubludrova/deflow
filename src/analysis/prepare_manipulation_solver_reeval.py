"""Bundle supplied checkpoints, runtime sources, and evaluation settings."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.manifest.read_text())
    if 'env_seeds' in selection or 'policy_seeds' in selection:
        raise ValueError('Use evaluation counts in the selection manifest')
    seed = 1
    environments = int(selection.get('environments', 20))
    samples = int(selection.get('samples_per_environment', 5))
    if environments < 1 or samples < 1:
        raise ValueError('Evaluation counts must be positive')
    env_seeds = list(range(seed, seed + environments))
    policy_seeds = list(range(seed, seed + samples))
    records = selection['jobs']
    keys = [r['key'] for r in records]
    if not keys or len(keys) != len(set(keys)) or any(Path(k).name != k or k in ('.', '..') for k in keys):
        raise ValueError('Job keys must be unique directory names')
    bundle = args.out / 'bundle'
    bundle.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source, bundle / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    jobs = []
    for row in records:
        paths = {k: args.manifest.parent / row[k] for k in ('checkpoint', 'contract', 'evaluation', 'engine')}
        old = json.loads(paths['evaluation'].read_text())
        config = json.loads(paths['contract'].read_text())
        assert sha(paths['checkpoint']) == old['checkpoint_sha256']
        assert sha(paths['engine']) == old['engine_module_sha256']
        assert old['evaluator_module_sha256'] == sha(bundle / 'source/analysis/evaluate_stage1_panel.py')
        dest = bundle / 'inputs' / row['key']
        dest.mkdir(parents=True)
        for key, name in [('checkpoint', 'checkpoint.pt'), ('contract', 'contract.json'),
                          ('evaluation', 'historical.json'), ('engine', 'engine.py')]:
            shutil.copy2(paths[key], dest / name)
        jobs.append(dict(key=row['key'], task=row['task'], seed=config['configuration']['seed'],
            run_id=config['run_id'], checkpoint_step=old['checkpoint_global_step'],
            checkpoint_sha256=old['checkpoint_sha256'], contract_sha256=sha(paths['contract']),
            training_engine_sha256=config['actor_module_sha256'],
            evaluation_engine_sha256=old['engine_module_sha256'],
            original_constructor=old['actor_constructor'], horizon=old['horizon']))
    manifest = dict(schema_version=1, jobs=jobs,
        arms={'original': {'max_iter_fwd': 25, 'tol_fwd': 1e-5},
              'tight100': {'max_iter_fwd': 100, 'tol_fwd': 1e-6}},
        env_seeds=env_seeds, policy_seeds=policy_seeds,
        episodes_per_job_per_arm=len(env_seeds) * (len(policy_seeds) + 1),
        max_job_seconds=selection.get('max_job_seconds', 10800),
        source_hashes={str(p.relative_to(bundle)): sha(p) for p in bundle.rglob('*.py')})
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'jobs': len(jobs), 'bundle': str(bundle)}))


if __name__ == '__main__':
    main()
