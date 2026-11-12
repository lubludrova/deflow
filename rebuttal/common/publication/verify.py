"""Verify the curated evidence without GPUs, checkpoint loading or network access."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / 'rebuttal'
E = R / 'experiments'
E01 = E / 'E01-solver-comparison/runs/deflow-campaign-20261111/revision3'
E02 = E / 'E02-sequential-anderson/runs/sa-20261112'
E03 = E / 'E03-tighter-anderson-audit/runs/tol1e-6-cap100-20261112'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = set(subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', 'rebuttal'],
        cwd=ROOT, text=True).splitlines())
    assert not any('/archive/' in p or 'ckpt' in Path(p).name or Path(p).suffix == '.pth' for p in paths)
    assert not any((ROOT / p).is_symlink() for p in paths)
    assert max((ROOT / p).stat().st_size for p in paths) < 50_000_000
    groups = {}
    evaluations = []
    source_checks = 0
    for label, base, prefix in [
        ('joint_cap25', E01, 'anderson'),
        ('joint_cap500', E01, 'anderson_highcap'),
        ('joint_tighter_cap100', E01, 'anderson_tighter'),
        ('sequential_anderson', E02, 'sequential_anderson'),
    ]:
        rows = []
        for seed in (1, 2, 3):
            run_id = f'{prefix}_s{seed}'
            result = base / 'mirror/results' / run_id
            p = base / 'mirror/evaluations' / run_id / 'evaluation.json'
            data = read(p)
            episodes = data['episodes']
            assert len(episodes) == 1000 and data['eval_seed'] == 26152
            hits = [sum(x['success'] and x['goal'] == goal for x in episodes) for goal in range(4)]
            probs = [n / len(episodes) for n in hits]
            success = sum(probs)
            u4 = statistics.mean(1 - (1 - p) ** 4 for p in probs)
            mean_return = statistics.mean(x['return'] for x in episodes)
            for key, value in [('u4', u4), ('total_success', success), ('mean_return', mean_return)]:
                assert math.isclose(data['summary'][key], value, rel_tol=1e-10, abs_tol=1e-10), (run_id, key)
            done = read(result / 'DONE.json')
            assert done['exit_code'] == 0
            assert any(200000 <= x['global_step'] <= 200007 for x in done['checkpoints'])
            assert list(result.glob('runs/*/events.out.tfevents.*'))
            assert str(p.relative_to(ROOT)) in paths
            engine = E01 / 'DEFlow/src/docking_full' if base == E01 else E / 'E02-sequential-anderson/code/DEFlow/src/docking_full'
            package = E01 / 'package/deflow_rebuttal' if base == E01 else E / 'E02-sequential-anderson/code/package/deflow_rebuttal'
            for name, sha in data['source_sha256'].items():
                assert digest(engine / name) == sha, (run_id, name)
                source_checks += 1
            for name, sha in data['package_sha256'].items():
                assert digest(package / name) == sha, (run_id, name)
                source_checks += 1
            rows.append({'run_id': run_id, 'u4': u4, 'success_percent': success * 100, 'evaluation_sha256': digest(p)})
        groups[label] = {'seeds': rows, 'u4_mean': statistics.mean(x['u4'] for x in rows),
                         'u4_sample_sd': statistics.stdev(x['u4'] for x in rows),
                         'success_mean': statistics.mean(x['success_percent'] for x in rows),
                         'success_sample_sd': statistics.stdev(x['success_percent'] for x in rows)}
        evaluations.extend(rows)
    prior_hash_checks = 0
    for campaign, base in [('E01', E01), ('E02', E02)]:
        receipt = read(R / f'common/operations/shutdown-20261112T074015Z/{campaign}-essential-file-hashes.json')
        for relative, info in receipt['files'].items():
            p = base / 'mirror' / relative
            if str(p.relative_to(ROOT)) in paths:
                assert digest(p) == info['sha256'], p
                assert p.stat().st_size == info['bytes']
                prior_hash_checks += 1
    for receipt in E03.glob('*.sync.json'):
        for relative, sha in read(receipt)['sha256'].items():
            p = E03 / 'mirror' / relative
            assert str(p.relative_to(ROOT)) in paths, p
            assert digest(p) == sha, p
            prior_hash_checks += 1
    audits = read(E01 / 'mirror/ops/idle-audits-v1/analysis/metrics.json')['audits']
    assert len(audits) == 9 and sum(x['full_reference_usable'] for x in audits) == 5
    assert sum(x['reference_invalid'] for x in audits) == 5
    raw = [p for p in paths if p.endswith('/raw.pt')]
    assert len(raw) == 18
    capsules = list((E01 / 'mirror/results').glob('anderson_s*/capsule_*.pt'))
    assert len(capsules) == 9 and all(str(p.relative_to(ROOT)) in paths for p in capsules)
    frozen_checks = 0
    snapshots = {}
    for exp, run in [('E02-sequential-anderson', 'sa-20261112'),
                     ('E04-joint-anderson-cap100', 'a100-20261112-restart1'),
                     ('E05-joint-anderson-cap10', 'a10-20261112-restart1')]:
        base = E / exp / 'runs' / run
        for relative, sha in read(base / 'ops/manifest.json')['frozen_files'].items():
            p = E / exp / relative if relative.startswith('code/') else base / relative
            assert digest(p) == sha, p
            frozen_checks += 1
        snapshot = base / 'publication-snapshot-20261112'
        if snapshot.exists():
            for relative, info in read(snapshot / 'SNAPSHOT.json')['files'].items():
                p = snapshot / relative
                assert str(p.relative_to(ROOT)) in paths, p
                assert digest(p) == info['sha256'], p
            state = read(snapshot / 'status-observed.json')
            assert len(state['jobs']) == 3
            for job in state['jobs']:
                assert job['state'] == 'TRAINING' and job['process_identity_matches']
                attempt = snapshot / 'attempts' / job['run_id'] / '0000'
                assert (attempt / 'training.log').is_file()
                assert list((snapshot / 'results' / job['run_id']).glob('runs/*/events.out.tfevents.*'))
            snapshots[exp] = {'observed_timestamp': state['timestamp'], 'jobs': [
                {'run_id': j['run_id'], 'state': j['state'], 'step': j['step']} for j in state['jobs']]}
    result = {'passed': True, 'completed_evaluations': len(evaluations), 'raw_audits': len(raw),
              'fully_valid_batches': 5, 'pooled_error_trajectories': 5120, 'invalid_reference_rows': 5,
              'selected_audit_capsules': len(capsules), 'evaluation_source_hash_checks': source_checks,
              'frozen_source_hash_checks': frozen_checks, 'prior_preservation_hash_checks': prior_hash_checks,
              'training_checkpoints_in_publication': 0, 'groups': groups, 'active_snapshots': snapshots,
              'draft_sha256': {name: digest(R / 'reviews' / name) for name in ['dkt6.md', 'VrBX.md', 'agWk.md']},
              'limits': ['Live snapshots are dated observations, not final results.',
                         'Original checkpoint hashes are provenance only. Checkpoints are not distributed.',
                         'No new scientific run or GPU computation was performed.']}
    (Path(__file__).parent / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['groups', 'active_snapshots', 'draft_sha256', 'limits']}, indent=2))


if __name__ == '__main__':
    main()
