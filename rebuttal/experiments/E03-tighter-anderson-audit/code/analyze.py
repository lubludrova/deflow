"""Recompute the three-row table and verify paired reference identity from saved tensors."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific(value):
    return f'{value:.2e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def main():
    root = Path(sys.argv[1])
    manifest = json.loads((root / 'manifest.json').read_text())
    source = Path(manifest['source_root'])
    rows, checks, provenance = [], [], {}
    pooled = {name: [] for name in ('1e-5 / 25', '1e-5 / 500', '1e-6 / 100')}
    for job in manifest['jobs']:
        new_path, old_path = Path(job['output']), Path(job['source_audit'])
        new_s = json.loads((new_path / 'summary.json').read_text())
        old_s = json.loads((old_path / 'summary.json').read_text())
        new = torch.load(new_path / 'raw.pt', map_location='cpu', weights_only=False)
        old = torch.load(old_path / 'raw.pt', map_location='cpu', weights_only=False)
        assert new_s['hashes'] == old_s['hashes']
        assert new_s['package_sha256'] == old_s['package_sha256']
        assert new_s['runtime_precision'] == old_s['runtime_precision']
        assert new_s['device_name'] == old_s['device_name']
        assert new_s['manifest']['repeats'] == old_s['manifest']['repeats'] == 10
        assert new_s['manifest']['warmup'] == old_s['manifest']['warmup'] == 2
        assert new_s['batch'] == old_s['batch'] == 1024
        assert new_s['loss_kind'] == old_s['loss_kind'] == 'configured_SAC_actor_loss'
        assert new_s['effective_anderson'] == {'tol': 1e-6, 'cap': 100}
        for key in ('obs', 'z'):
            assert torch.equal(new[key], old[key]), key
        reference_checks = {}
        for key, tensor in old['reference'].items():
            other = new['reference'][key]
            same = (torch.allclose(tensor, other, rtol=0, atol=0, equal_nan=True)
                    if tensor.is_floating_point() else torch.equal(tensor, other))
            reference_checks[key] = same
            assert same, (job['run_id'], 'reference', key)
        ref = new.get('reference_readout')
        if ref is not None:
            for key, tensor in ref.items():
                assert torch.equal(tensor, old['reference_readout'][key]), key
        assert (ref is not None) == bool(new['reference']['valid'].all())
        checks.append(dict(run_id=job['run_id'], actual_step=job['actual_step'],
                           reference_valid=new_s['reference_valid'],
                           reference_invalid=new_s['reference_invalid'],
                           reference_tensors_identical=reference_checks,
                           reference_readout_identical=ref is not None))
        for setting, raw, summary, label in [
                ('1e-5 / 25', old, old_s, 'anderson_cap25'),
                ('1e-5 / 500', old, old_s, 'anderson_cap500'),
                ('1e-6 / 100', new, new_s, 'anderson')]:
            row = dict(run_id=job['run_id'], seed=job['seed'], actual_step=job['actual_step'],
                       setting=setting, forward_ms=summary[label]['forward_seconds_median']*1000,
                       reference_valid=summary['reference_valid'],
                       reference_invalid=summary['reference_invalid'],
                       native_convergence=summary[label]['convergence'],
                       absolute_actor_loss_error=None, gradient_relative_error=None)
            if ref is not None:
                rd = raw[label]['readout']
                row['absolute_actor_loss_error'] = abs(float(rd['loss'])-float(ref['loss']))
                row['signed_actor_loss_error'] = float(rd['loss'])-float(ref['loss'])
                row['gradient_relative_error'] = float(
                    (rd['gradient'].double()-ref['gradient']).norm()/ref['gradient'].norm())
                pooled[setting].append((rd['logp'].double()-ref['logp']).abs().reshape(-1))
            rows.append(row)
        for folder in (old_path, new_path):
            for name in ('summary.json', 'raw.pt'):
                provenance[str(folder/name)] = digest(folder/name)
    aggregate = {}
    training = {}
    for setting, arm in [('1e-5 / 25', 'anderson'), ('1e-5 / 500', 'anderson_highcap'),
                         ('1e-6 / 100', 'anderson_tighter')]:
        subset = [row for row in rows if row['setting'] == setting]
        valid = [row for row in subset if row['absolute_actor_loss_error'] is not None]
        logp = torch.cat(pooled[setting])
        values = []
        for seed in (1, 2, 3):
            p = source/'evaluations'/f'{arm}_s{seed}'/'evaluation.json'
            evaluation = json.loads(p.read_text())
            episodes = evaluation['episodes']
            assert len(episodes) == 1000 and evaluation['eval_seed'] == 26152
            values.append(dict(seed=seed, success=100*statistics.mean(x['success'] for x in episodes),
                               mean_return=statistics.mean(x['return'] for x in episodes)))
            provenance[str(p)] = digest(p)
        training[setting] = values
        aggregate[setting] = dict(forward_ms=statistics.median(x['forward_ms'] for x in subset),
            p95_absolute_log_density_error=float(torch.quantile(logp, .95)),
            median_absolute_actor_loss_error=statistics.median(x['absolute_actor_loss_error'] for x in valid),
            gradient_relative_error_median=statistics.median(x['gradient_relative_error'] for x in valid),
            gradient_relative_error_range=[min(x['gradient_relative_error'] for x in valid),
                                           max(x['gradient_relative_error'] for x in valid)],
            native_nonconverged_cases=sum(not x['native_convergence']['mean_criterion_met'] for x in subset),
            timing_cases=len(subset), error_cases=len(valid), logp_trajectories=logp.numel(),
            success_mean=statistics.mean(x['success'] for x in values),
            success_sample_sd=statistics.stdev(x['success'] for x in values),
            return_mean=statistics.mean(x['mean_return'] for x in values),
            return_sample_sd=statistics.stdev(x['mean_return'] for x in values))
    result = dict(aggregate=aggregate, per_case=rows, reference_checks=checks,
                  independent_training=training, source_sha256=provenance,
                  analysis_script_sha256=digest(__file__))
    out = root/'analysis'
    out.mkdir(exist_ok=False)
    (out/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Completed tighter-Anderson audit', '',
             '| Anderson tolerance / cap | Forward ms/batch | P95 absolute log-density error (nats) | Median absolute actor-loss error | Success (%) | Return |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for setting, v in aggregate.items():
        lines.append(f"| {setting} | {v['forward_ms']:.2f} | {scientific(v['p95_absolute_log_density_error'])} | "
                     f"{scientific(v['median_absolute_actor_loss_error'])} | "
                     f"{v['success_mean']:.2f} ± {v['success_sample_sd']:.2f} | "
                     f"{v['return_mean']:.2f} ± {v['return_sample_sd']:.2f} |")
    lines.extend(['', 'Timing and errors compare solver settings on the same nine saved A25-policy batches.',
                  'Success and return are separate training results, mean ± sample SD over three seeds,',
                  'with 1,000 stochastic episodes per policy using its training solver.',
                  'Timing: RTX 4090, median of nine case medians, each with ten repetitions after two warmups.',
                  'Includes solver diagnostics, excludes density/readout/backward and training telemetry.',
                  'Errors: five fully valid FP64-reference batches. Log-density P95 pools 5,120 trajectories.',
                  'Five reference-invalid trajectories affect four batches: all three near 100k and seed 3 near 200k.',
                  'Those full-batch errors remain unavailable. All reference tensors and valid reference readouts',
                  'match E01 exactly. New timing was measured in a separate exclusive session on the same GPU.',
                  'Small scalar loss error does not certify gradient accuracy. Safeguarded equilibrium agreement',
                  'does not certify exact finite-iteration sampler densities or sensitivities.', ''])
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
