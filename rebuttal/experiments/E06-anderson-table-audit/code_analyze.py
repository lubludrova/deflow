"""Recompute every displayed table cell from immutable episode/tensor evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(a, b):
    return bool(torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)) if a.is_floating_point() else bool(torch.equal(a, b))


def main(root):
    m = json.loads((root / 'manifest.json').read_text())
    settings = ['1e-5_10', '1e-5_25', '1e-5_500', '1e-5_100', '1e-6_100']
    rows = {key: [] for key in settings}
    pooled = {key: [] for key in settings}
    reference_checks, sources = [], {}
    for job in m['jobs']:
        out = root / 'audits' / job['run_id']
        summary = json.loads((out / 'summary.json').read_text())
        raw = torch.load(out / 'raw.pt', map_location='cpu', weights_only=False)
        baseline_path = root / 'baseline' / job['run_id'].rsplit('_', 1)[0]
        baseline = torch.load(baseline_path / 'raw.pt', map_location='cpu', weights_only=False)
        old_s = json.loads((baseline_path / 'summary.json').read_text())
        assert summary['hashes'] == old_s['hashes']
        assert {k: v for k, v in summary['package_sha256'].items() if not k.startswith('._')} == old_s['package_sha256']
        assert summary['batch'] == 1024 and summary['manifest']['repeats'] == 10 and summary['manifest']['warmup'] == 2
        assert summary['device_name'] == 'NVIDIA GeForce RTX 5090'
        assert summary['runtime_precision']['matmul_allow_tf32'] is False
        assert summary['runtime_precision']['cudnn_allow_tf32'] is False
        assert summary['loss_kind'] == 'configured_SAC_actor_loss'
        assert all(torch.equal(raw[k], baseline[k]) for k in ('obs', 'z'))
        assert torch.equal(raw['reference']['valid'], baseline['reference']['valid'])
        refcheck = {'run_id': job['run_id'], 'valid_mask_identical': True, 'invalid': summary['reference_invalid'], 'fields': {}}
        for k, v in raw['reference'].items():
            other = baseline['reference'][k]
            finite = torch.isfinite(v) & torch.isfinite(other)
            difference = float((v[finite]-other[finite]).abs().max()) if v.is_floating_point() and finite.any() else None
            refcheck['fields'][k] = {'bitwise_equal': same(v, other), 'maximum_absolute_difference': difference}
        ref = raw.get('reference_readout')
        assert (ref is not None) == bool(raw['reference']['valid'].all())
        if ref is not None:
            refcheck['readout_max_absolute_differences'] = {k: float((v-baseline['reference_readout'][k]).abs().max()) for k,v in ref.items()}
            # FP64 hardware/runtime rounding is allowed, but a changed branch is not silently accepted.
            assert torch.allclose(raw['reference']['trajectory'], baseline['reference']['trajectory'], rtol=1e-10, atol=1e-10)
            for k,v in ref.items():
                assert torch.allclose(v, baseline['reference_readout'][k], rtol=1e-9, atol=1e-10), (job['run_id'], k)
        reference_checks.append(refcheck)
        for cap in job['caps']:
            key = ('1e-6_' if job['tol'] == 1e-6 else '1e-5_') + str(cap)
            label = 'anderson' if job['tol'] == 1e-6 else f'anderson_cap{cap}'
            s = summary[label]
            assert len(s['forward_seconds_repetitions']) == 10
            row = {'run_id': job['run_id'], 'forward_ms': 1000*s['forward_seconds_median'],
                   'reference_valid': summary['reference_valid'], 'gradient_percent': None}
            if ref is not None:
                rd = raw[label]['readout']
                assert torch.isfinite(rd['gradient']).all() and torch.isfinite(rd['logp']).all()
                gradient = 100*float((rd['gradient'].double()-ref['gradient']).norm()/ref['gradient'].norm())
                assert abs(gradient-100*s['gradient_relative_difference']) < 1e-9
                row['gradient_percent'] = gradient
                pooled[key].append((rd['logp'].double()-ref['logp']).abs().reshape(-1))
            rows[key].append(row)
        for p in (out/'raw.pt', out/'summary.json', baseline_path/'raw.pt', baseline_path/'summary.json'):
            sources[str(p.relative_to(root))] = sha(p)
        attempt = root/'attempts'/job['run_id']
        assert json.loads((attempt/'EXIT.json').read_text())['exit_code'] == 0
        ack = json.loads((attempt/'SYNC_ACK.json').read_text())
        for rel,h in ack['sha256'].items():
            assert sha(root/rel) == h
    metrics = {}
    for key in settings:
        gradients = torch.tensor([r['gradient_percent'] for r in rows[key] if r['gradient_percent'] is not None],dtype=torch.float64)
        errors = torch.cat(pooled[key])
        assert len(rows[key]) == 9 and len(gradients) == 5 and errors.numel() == 5120
        success = []
        for seed in (1,2,3):
            p = root/'evaluations'/f'{key}_s{seed}.json'
            evaluation = json.loads(p.read_text())
            assert len(evaluation['episodes']) == 1000 and evaluation['eval_seed'] == 26152
            rate = statistics.mean(float(e['success']) for e in evaluation['episodes'])*100
            assert abs(rate-evaluation['summary']['total_success']*100) < 1e-10
            success.append(rate)
            sources[str(p.relative_to(root))] = sha(p)
        metrics[key] = {'forward_ms': statistics.median(r['forward_ms'] for r in rows[key]),
                        'log_density_median': float(torch.quantile(errors,.5)),
                        'log_density_p95': float(torch.quantile(errors,.95)),
                        'gradient_percent_median': float(torch.quantile(gradients,.5)),
                        'gradient_percent_p95': float(torch.quantile(gradients,.95)),
                        'success_mean': statistics.mean(success), 'success_sample_sd': statistics.stdev(success),
                        'success_by_seed': success, 'timing_batches': 9, 'error_batches': 5, 'log_density_trajectories': 5120}
    result = {'metrics': metrics, 'per_case': rows, 'reference_checks': reference_checks,
              'source_sha256': sources, 'analysis_sha256': sha(Path(__file__)),
              'protocol': 'Same nine E01 A25 policy inputs. RTX5090/torch2.7.1+cu128 strict FP32, same source. Median of nine case timing medians (10 repetitions, 2 warmups). Absolute log-density errors pooled over 5120 trajectories from five fully valid FP64-reference batches. Gradient relative L2 error in percent, median and linearly interpolated P95 over those five batch gradients. Success mean and sample SD across three independent trained seeds, each with 1000 stochastic episodes, eval seed26152. No filtering of training seeds or evaluation episodes.',
              'limitations': 'Numerical errors concern frozen A25-trained inputs, not independently trained policies. Four reference-invalid batches remain preserved and excluded from all full-batch error statistics. Errors compare safeguarded equilibrium readouts, not exact finite-program likelihood/derivatives. FP64 cross-runtime roundoff is recorded. Training success cohorts use their original hardware/runtime.'}
    analysis = root/'analysis'
    analysis.mkdir(exist_ok=True)
    with (analysis/'metrics.json').open('x') as f:json.dump(result,f,indent=2)
    lines = ['| Anderson setting (tolerance / cap) | Forward time (ms/batch, RTX 5090) | Log-density (nat), Median | Log-density (nat), P95 | Gradient (%), Median | Gradient (%), P95 | Success (%) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key in settings:
        v=metrics[key]
        lines.append(f"| {key.replace('_', ' / ')} | {v['forward_ms']:.2f} | {v['log_density_median']:.2e} | {v['log_density_p95']:.2e} | {v['gradient_percent_median']:.3g} | {v['gradient_percent_p95']:.3g} | {v['success_mean']:.2f} ± {v['success_sample_sd']:.2f} |")
    with (analysis/'TABLE.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    with (analysis/'METHOD.md').open('x') as f:f.write(result['protocol']+'\n\n'+result['limitations']+'\n')
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
