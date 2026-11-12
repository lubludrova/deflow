"""Offline verification of completed cap10/cap100 evidence, without checkpoints."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / 'rebuttal/experiments'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main():
    results = {}
    checked = 0
    for cap, number in [(100, 4), (10, 5)]:
        exp = EXPERIMENTS / f'E0{number}-joint-anderson-cap{cap}'
        run = exp / f'runs/a{cap}-20261112-restart1'
        snapshot = run / 'completion-snapshot-20261112'
        for relative, info in read(snapshot/'SNAPSHOT.json')['files'].items():
            p = snapshot/relative
            assert p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], p
            checked += 1
        expected = read(run/'analysis/metrics.json')
        observed = []
        for seed in (1, 2, 3):
            rid = f'joint_anderson_a{cap}_s{seed}'
            p = snapshot/'evaluations'/rid/'evaluation.json'
            ev = read(p)
            assert len(ev['episodes']) == 1000 and ev['eval_seed'] == 26152
            for name, digest in ev['source_sha256'].items():
                assert sha(exp/'code/DEFlow/src/docking_full'/name) == digest
            for name, digest in ev['package_sha256'].items():
                assert sha(exp/'code/package/deflow_rebuttal'/name) == digest
            success = statistics.mean(e['success'] for e in ev['episodes'])
            probs = [sum(e['success'] and e['goal'] == g for e in ev['episodes'])/1000 for g in range(4)]
            u4 = statistics.mean(1-(1-p)**4 for p in probs)
            ret = statistics.mean(e['return'] for e in ev['episodes'])
            attempt = snapshot/'attempts'/rid/'0000'
            assert read(attempt/'training_exit.json')['exit_code'] == 0
            accepted = read(attempt/'TRAINING_ACCEPTED.json')
            assert 200000 <= accepted['step'] <= 200007
            assert ev['checkpoint_sha256'] == accepted['sha256']
            eval_attempt = snapshot/'ops/evaluation-handoff-v1/attempts'/rid/'0000' if cap == 10 and seed in (1,3) else attempt
            assert read(eval_attempt/'evaluation_exit.json')['exit_code'] == 0
            assert read(eval_attempt/'EVALUATION_ACCEPTED.json')['sha256'] == sha(p)
            done = read(snapshot/'results'/rid/'DONE.json')
            values = {'u4': u4, 'success_percent': 100*success, 'mean_return': ret,
                      'training_sps': accepted['step']/done['elapsed_seconds']}
            for key, value in values.items():close(value, expected['seeds'][seed-1][key])
            observed.append(values)
        for key in observed[0]:
            close(statistics.mean(r[key] for r in observed), expected['mean_and_sample_sd'][key]['mean'])
            close(statistics.stdev(r[key] for r in observed), expected['mean_and_sample_sd'][key]['sample_sd'])
        results[str(cap)] = expected['mean_and_sample_sd']
    exp = EXPERIMENTS/'E06-anderson-table-audit'
    run = exp/'runs/rtx5090-20261112'
    analysis = read(exp/'analysis/metrics.json')
    for relative, digest in analysis['source_sha256'].items():
        path = run/'mirror'/relative if relative.startswith('audits/') else run/relative
        assert sha(path) == digest, path
        checked += 1
    assert read(exp/'analysis/INDEPENDENT_VERIFICATION.json')['metrics_sha256'] == sha(exp/'analysis/metrics.json')
    manifest = read(run/'manifest.json')
    for relative, digest in manifest['frozen_files'].items():assert sha(run/relative) == digest
    for job in manifest['jobs']:
        assert sha(run/job['input']) == job['input_sha256']
        assert read(run/'mirror/attempts'/job['run_id']/'EXIT.json')['exit_code'] == 0
    assert len(manifest['jobs']) == 18
    for key, values in analysis['metrics'].items():
        assert values['timing_batches'] == 9 and values['error_batches'] == 5 and values['log_density_trajectories'] == 5120
        success = [100*statistics.mean(e['success'] for e in read(run/'evaluations'/f'{key}_s{s}.json')['episodes']) for s in (1,2,3)]
        close(statistics.mean(success), values['success_mean'])
        close(statistics.stdev(success), values['success_sample_sd'])
        cases = analysis['per_case'][key]
        close(statistics.median(c['forward_ms'] for c in cases), values['forward_ms'])
        gradients = sorted(c['gradient_percent'] for c in cases if c['gradient_percent'] is not None)
        close(statistics.median(gradients), values['gradient_percent_median'])
        close(gradients[3]*0.2+gradients[4]*0.8, values['gradient_percent_p95'])
    draft = (ROOT/'rebuttal/reviews/dkt6.md').read_text()
    start = draft.index('| Anderson setting (tolerance / cap) |')
    table = draft[start:draft.index('\n\n', start)].splitlines()[2:]
    for line in table:
        cells = [c.strip() for c in line.strip('|').split('|')]
        key = cells[0].replace(' / ', '_')
        v = analysis['metrics'][key]
        expected = [f"{v['forward_ms']:.2f}", f"{v['log_density_median']:.2e}", f"{v['log_density_p95']:.2e}", f"{v['gradient_percent_median']:.3g}", f"{v['gradient_percent_p95']:.3g}"]
        assert [float(x) for x in cells[1:6]] == [float(x) for x in expected]
        assert cells[6] == f"{v['success_mean']:.2f} ± {v['success_sample_sd']:.2f}"
    stopping = read(exp/'analysis/EARLY_STOPPING.json')
    for row in stopping['rows']:
        for setting, expected in row['settings'].items():
            tol, cap = setting.split('/')
            kind = 'sweep' if tol == '1e-5' else 'tight'
            label = f'anderson_cap{cap}' if kind == 'sweep' else 'anderson'
            actual = read(run/f"mirror/audits/s{row['seed']}_{row['target_step']}_{kind}/summary.json")[label]['convergence']
            assert all(actual[k] == v for k,v in expected.items())
    inventory = Path(__file__).parent/'files.sha256'
    if inventory.exists():
        for line in inventory.read_text().splitlines():
            digest, relative = line.split('  ', 1)
            assert sha(ROOT/relative) == digest, relative
    print(json.dumps({'passed': True, 'evaluation_episodes_recomputed': 6000,
                      'audit_exit_codes_checked': 18, 'evidence_hashes_checked': checked,
                      'aggregate': results}, indent=2))


if __name__ == '__main__':
    main()
