"""Derive the rebuttal audit table without manufacturing invalid reference errors."""
import csv
import json
import math
from pathlib import Path
import statistics

LABELS = ('anderson_cap25', 'anderson_cap100', 'anderson_cap500', 'newton')


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def metric_row(summary, label, reference_usable, reference_loss):
    source = summary[label]
    loss = source.get('loss')
    p95 = source.get('logp_abs_difference', {}).get('p95')
    gradient = source.get('gradient_relative_difference')
    return dict(solver=label, forward_ms=1000*source['forward_seconds_median'],
        reference_valid=summary['reference_valid'], reference_invalid=summary['reference_invalid'],
        full_reference_usable=reference_usable,
        p95_logp_error=p95 if reference_usable and finite(p95) else None,
        gradient_relative_error=gradient if reference_usable and finite(gradient) else None,
        gradient_reference_norm=source.get('gradient_reference_norm') if reference_usable else None,
        delta_actor_loss=loss-reference_loss if reference_usable and finite(loss) and finite(reference_loss) else None,
        candidate_loss=loss, candidate_readout_error=source.get('readout_error'),
        residual_p95=source.get('local_absolute', {}).get('p95'),
        residual_max=source.get('local_absolute', {}).get('max'))


def median(values):
    values = [v for v in values if finite(v)]
    return statistics.median(values) if values else None


def build_tables(root, operations):
    import torch
    root, operations = Path(root), Path(operations)
    report = json.loads((operations / 'campaign_report.json').read_text())
    rows, audits = [], []
    pooled = {label: [] for label in LABELS}
    for name, state in report['audits'].items():
        audit = dict(audit=name, state=state['state'], seed=state['seed'],
            target_step=state['target_step'], actual_step=state.get('actual_step'))
        if state['state'] != 'FINISHED':
            audits.append(audit)
            continue
        output = root / 'audits' / name
        summary = json.loads((output / 'summary.json').read_text())
        raw = torch.load(output / 'raw.pt', map_location='cpu', weights_only=False)
        valid = raw['reference']['valid']
        if valid.numel() != 1024 or int(valid.sum()) != summary['reference_valid']:
            raise RuntimeError('Reference mask/summary mismatch: ' + name)
        reference = raw.get('reference_readout')
        usable = bool(valid.all()) and reference is not None
        if usable:
            usable = all(torch.isfinite(reference[k]).all().item()
                         for k in ('action', 'logp', 'loss', 'gradient'))
        reference_loss = float(reference['loss']) if usable else None
        audit.update(reference_valid=summary['reference_valid'], reference_invalid=summary['reference_invalid'],
            full_reference_usable=usable, reference_loss=reference_loss,
            reference_readout_error=summary.get('reference_readout_error'))
        audits.append(audit)
        for label in LABELS:
            row = dict(audit=name, seed=state['seed'], target_step=state['target_step'],
                actual_step=state['actual_step'], **metric_row(summary, label, usable, reference_loss))
            rows.append(row)
            candidate = raw[label].get('readout')
            if usable and candidate is not None and torch.isfinite(candidate['logp']).all():
                errors = (candidate['logp'].double()-reference['logp'].double()).abs().flatten()
                pooled[label].append(errors)
    aggregates = {}
    for label in LABELS:
        selected = [row for row in rows if row['solver'] == label]
        errors = torch.cat(pooled[label]) if pooled[label] else None
        aggregates[label] = dict(
            forward_ms_median=median([row['forward_ms'] for row in selected]),
            logp_error_p95=float(torch.quantile(errors, .95)) if errors is not None else None,
            gradient_relative_error_median=median([row['gradient_relative_error'] for row in selected]),
            absolute_delta_actor_loss_median=median([abs(row['delta_actor_loss'])
                for row in selected if row['delta_actor_loss'] is not None]),
            timing_cases=len(selected), logp_trajectories=int(errors.numel()) if errors is not None else 0,
            gradient_cases=sum(row['gradient_relative_error'] is not None for row in selected),
            loss_cases=sum(row['delta_actor_loss'] is not None for row in selected))
    out = operations / 'analysis'
    out.mkdir(exist_ok=True)
    payload = dict(audits=audits, rows=rows, aggregate=aggregates,
        units=dict(forward_ms='synchronized solver-only host milliseconds per batch of 1024',
            logp_error='nats', gradient_relative_error='dimensionless L2 relative error',
            delta_actor_loss='candidate minus FP64 reference, saved configured SAC actor objective'),
        aggregation='Median across finished audit cases for timing/gradient/absolute loss error. Logp P95 pools trajectories only from usable complete-batch references. Explicit denominators accompany every solver.',
        limitations=['Invalid references and failed candidates remain in per-audit rows and raw.pt.',
            'No complete-batch gradient comparison when any reference row is invalid.',
            'A-policy capsule evidence does not cover all N-policy failure regimes.',
            'The safeguarded equilibrium reference does not certify finite-sampler density.',
            'A100 here has tol1e-5 and is distinct from the T training arm at tol1e-6.'])
    (out / 'metrics.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (out / 'per_audit.csv').open('w', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    def show(value):
        return 'N/A' if value is None else f'{value:.6g}'
    lines = ['# Paired solver audits', '',
        '| Solver | Median forward ms | P95 logp error | Median relative gradient error | Median absolute delta actor loss | Timing / gradient cases | Logp trajectories |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for label, item in aggregates.items():
        lines.append('| '+ ' | '.join([label,show(item['forward_ms_median']),show(item['logp_error_p95']),
            show(item['gradient_relative_error_median']),show(item['absolute_delta_actor_loss_median']),
            str(item['timing_cases'])+' / '+str(item['gradient_cases']),str(item['logp_trajectories'])])+' |')
    lines.extend(['', '| Audit | State | Target / actual step | Valid / invalid reference rows | Full reference usable |',
        '| --- | --- | --- | --- | --- |'])
    for row in audits:
        lines.append('| '+ ' | '.join(str(v) for v in [row['audit'],row['state'],
            str(row['target_step'])+' / '+str(row.get('actual_step','—')),
            str(row.get('reference_valid','—'))+' / '+str(row.get('reference_invalid','—')),
            row.get('full_reference_usable','—')])+' |')
    lines.extend(['',payload['aggregation'],'',*payload['limitations'],
        '', 'Signed per-case loss deltas, residual tails and all case denominators are in metrics.json and per_audit.csv.'])
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    return [out / name for name in ('metrics.json','per_audit.csv','REPORT.md')]
