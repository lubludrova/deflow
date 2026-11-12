"""Durable execution of the exact approved DEFlow queue, then paired audits."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import subprocess
import threading
import time
import traceback
from common import *

CAMPAIGN = json.loads((PACKAGE / 'night_campaign.json').read_text())
JOBS = {job['id']: job for job in CAMPAIGN['jobs']}
STATE = {key: dict(state='PLANNED', seed=job['seed'], gpu_slot=job['gpu_slot']) for key, job in JOBS.items()}
AUDITS = {f'audit_a{seed}_{target}': dict(state='PLANNED', seed=seed, target_step=target)
          for seed in (1, 2, 3) for target in (25000, 100000, 200000)}
REPORT_LOCK = threading.RLock()


def update(run_id, **values):
    with LOCK:
        STATE[run_id].update(values)


def train_slot(slot):
    previous = None
    for run_id in CAMPAIGN['queue_by_gpu_slot'][str(slot)]:
        job = JOBS[run_id]
        try:
            update(run_id, state='WAITING_FOR_SYNC')
            sync_gate(previous)
            contract_check()
            run = OUTPUT / run_id
            update(run_id, state='RUNNING', started_at=time.time())
            training = execute(run_id, train_argv(job, run), slot, run, scientific=True, required=previous)
            checkpoint = validate_run(run, 200000, 1024)
            accepted(run_id, [checkpoint, run / 'DONE.json', run / 'manifest.json', run / 'runtime_precision.json'])
            update(run_id, state='TRAINING_FINISHED', checkpoint=str(checkpoint), training=training)
            contract_check()
            update(run_id, state='EVALUATING')
            evaluation = evaluate(run_id + '_eval', run, checkpoint, ROOT / 'evaluations' / run_id, 1000, slot, scientific=True)
            metrics = json.loads((ROOT / 'evaluations' / run_id / 'evaluation.json').read_text())['summary']
            update(run_id, state='EVALUATION_FINISHED', evaluation=evaluation, metrics=metrics)
            previous = run_id + '_eval'
            report()
        except BaseException:
            error = traceback.format_exc()
            started = (OPS / 'attempts' / run_id / 'START.json').exists()
            state = ('INTERRUPTED' if stop_requested() else 'FAILED') if started else 'BLOCKED'
            update(run_id, state=state, error=error)
            write(OPS / 'failures' / (run_id + '.json'), dict(run_id=run_id, at=time.time(), traceback=error))
            queue = CAMPAIGN['queue_by_gpu_slot'][str(slot)]
            for pending in queue[queue.index(run_id) + 1:]:
                update(pending, state='BLOCKED', reason=f'Slot blocked by {run_id}')
            report()
            return


def select_capsules():
    import torch
    selection = []
    for seed in (1, 2, 3):
        run_id = f'anderson_s{seed}'
        run = OUTPUT / run_id
        candidates = []
        for path in sorted(run.glob('capsule_*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            metadata = payload.get('metadata', {})
            if (metadata.get('role') == 'training/actor_grad' and metadata.get('batch') == 1024
                    and metadata.get('global_step') is not None
                    and len(payload.get('obs', [])) == 1024
                    and {'actor', 'q1', 'q2', 'loss_config'} <= payload.keys()):
                candidates.append((int(metadata['global_step']), path))
        for target in (25000, 100000, 200000):
            row = dict(seed=seed, target_step=target, run_id=run_id, source_run_state=STATE[run_id]['state'],
                       rule='Nearest eligible full-batch training/actor_grad capsule. Ties: smaller actual step, then lexical path.')
            if candidates:
                step, path = min(candidates, key=lambda pair: (abs(pair[0] - target), pair[0], str(pair[1])))
                row.update(actual_step=step, path=str(path), sha256=digest(path), state='PLANNED')
            else:
                row.update(state='MISSING', reason='No eligible capsule. No substitute checkpoint used.')
            selection.append(row)
    write(OPS / 'audit_selection.json', selection)
    return selection


def audit_seed(seed, selection):
    slot = seed - 1
    for row in [r for r in selection if r['seed'] == seed]:
        run_id = f'audit_a{seed}_{row["target_step"]}'
        if row['state'] == 'MISSING':
            with LOCK:
                AUDITS[run_id] = row
            continue
        try:
            sync_gate()
            contract_check()
            path = row['path']
            if digest(path) != row['sha256']:
                raise RuntimeError('Selected capsule changed')
            output = ROOT / 'audits' / run_id
            result = execute(run_id, [PYTHON, '-m', 'deflow_rebuttal.audit', '--engine',
                str(REPO / 'src/docking_full/sac_anderson_flow.py'), '--checkpoint', path,
                '--inputs', path, '--anderson-caps', '25', '100', '500', '--anderson-tol', '1e-5',
                '--output', str(output), '--device', 'cuda', '--warmup', '2', '--repeats', '10',
                '--precision-policy', 'strict_fp32', '--require-same-capsule'], slot, scientific=True)
            summary = json.loads((output / 'summary.json').read_text())
            if (summary['batch'] != 1024 or summary['loss_kind'] != 'configured_SAC_actor_loss'
                    or summary.get('precision_policy') != 'strict_fp32'
                    or summary.get('input_kind') != 'campaign_same_capsule'
                    or summary.get('runtime_precision', {}).get('matmul_allow_tf32') is not False):
                raise RuntimeError('Audit batch or loss contract mismatch')
            if not (output / 'raw.pt').stat().st_size:
                raise RuntimeError('Missing audit raw artifact')
            accepted(run_id, [output / 'summary.json', Path(path)])
            with LOCK:
                AUDITS[run_id] = dict(row, state='FINISHED', receipt=result, summary=summary)
        except BaseException:
            with LOCK:
                AUDITS[run_id] = dict(row, state='FAILED', traceback=traceback.format_exc())
            write(OPS / 'failures' / (run_id + '.json'), AUDITS[run_id])
            with LOCK:
                for key, pending in AUDITS.items():
                    if pending['seed'] == seed and pending['state'] == 'PLANNED':
                        pending.update(state='BLOCKED', reason=f'Audit slot failed at {run_id}')
            report()
            break
        report()


def report():
    with REPORT_LOCK:
        _report()


def _report():
    with LOCK:
        rows = json.loads(json.dumps(STATE))
        audits = json.loads(json.dumps(AUDITS))
    telemetry = {}
    for run_id in rows:
        path = OUTPUT / run_id / 'telemetry_summary.json'
        if path.exists():
            try:
                telemetry[run_id] = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
    evaluation_telemetry = {}
    for run_id in rows:
        path = ROOT / 'evaluations' / run_id / 'telemetry_summary.json'
        if path.exists():
            try:
                evaluation_telemetry[run_id] = read(path)
            except json.JSONDecodeError:
                pass
    write(ROOT / 'campaign_report.json', dict(observed_at=time.time(), campaign_id=read(OPS / 'execution_contract.json')['campaign_id'],
        contract_sha256=os.environ.get('DEFLOW_CONTRACT_SHA256'), precision_policy='strict_fp32',
        counts={'runs': len(rows), 'evaluated': sum(r['state'] == 'EVALUATION_FINISHED' for r in rows.values()),
                'audits': len(audits), 'audits_finished': sum(r['state'] == 'FINISHED' for r in audits.values())},
        runs=rows, audits=audits, training_telemetry=telemetry, evaluation_telemetry=evaluation_telemetry))
    lines = ['# DEFlow solver campaign', '',
        'Training and evaluation are separate. All seeds, failures and deferred jobs are retained.', '',
        '| Run | State | U4 | Success | Return | Train seconds | Eval seconds |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for run_id, row in rows.items():
        m = row.get('metrics', {})
        values = [run_id, row['state'], m.get('u4', ''), m.get('total_success', ''), m.get('mean_return', ''),
                  row.get('training', {}).get('elapsed_seconds', ''), row.get('evaluation', {}).get('elapsed_seconds', '')]
        lines.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    lines += ['', '## Frozen paired audit', '',
        '| Audit | State | Target | Actual | Valid FP64 | Invalid FP64 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for key, row in audits.items():
        s = row.get('summary', {})
        lines.append('| ' + ' | '.join(str(v) for v in [key, row['state'], row['target_step'], row.get('actual_step', ''),
            s.get('reference_valid', ''), s.get('reference_invalid', '')]) + ' |')
    lines += ['', 'Full solver curves, per-example tails, paired actions/logp/SAC gradients and reference exclusions:',
        '`audits/*/summary.json`, `audits/*/raw.pt`. Per-role training/evaluation denominators:',
        '`campaign_report.json` and each `telemetry_summary.json`. Raw all-call telemetry stays on the server.', '',
        'A/H changes the iteration budget. A/T changes both budget and tolerance. Cap-reaching and convergence failure are separate.',
        'Cap500 is not a convergence certificate. This is neither exact finite-sampler likelihood certification nor an equal-wall-clock SAC/DIME comparison.']
    with LOCK:
        (ROOT / 'campaign_report.md').write_text('\n'.join(lines) + '\n')


def monitor(stop):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    history = {}
    resource_window = []
    while not stop.is_set():
        now = time.time()
        sample = gpu_sample()
        resource_window.append(sample)
        resource_window = [s for s in resource_window if now - s['at'] <= 600]
        with (OPS / 'resource_history.jsonl').open('a') as stream:
            stream.write(json.dumps(sample) + '\n')
        with LOCK:
            active = json.loads(json.dumps(ACTIVE))
            jobs = json.loads(json.dumps(STATE))
        for run_id, contract in active.items():
            run = Path(contract['run_path']) if contract.get('run_path') else ROOT / 'audits' / run_id
            telemetry = progress(run)
            step = telemetry['step']
            observations = history.setdefault(run_id, [])
            if step > 0 and (not observations or observations[-1][1] != step):
                observations.append((now, step))
            recent = [row for row in observations if now - row[0] <= 600]
            if len(recent) >= 2 and recent[-1][0] - recent[0][0] >= 60:
                rate = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
                telemetry['delta_sps'] = rate
                telemetry['eta_seconds'] = max(0, 200000 - step) / rate if rate > 0 and run_id in JOBS else None
                telemetry['eta_caveat'] = 'Measured recent window. Late solver cost and final evaluation may increase duration.'
            uuid = contract['gpu_uuid']
            matching = [r for s in resource_window for r in s['gpus'] if r[1] == uuid]
            utilization = sorted(int(row[4]) for row in matching)
            telemetry['gpu_utilization_window'] = dict(samples=len(utilization),
                mean=sum(utilization) / len(utilization), p95=utilization[int(.95 * (len(utilization) - 1))],
                peak=max(utilization), memory_used_mib=int(matching[-1][2]))
            telemetry['co_runners'] = [r for r in sample['processes'] if r[0] == uuid and int(r[1]) != contract['pid']]
            checkpoints = list(run.glob('runs/*/*.pt'))
            telemetry['checkpoint_age_seconds'] = now - max(p.stat().st_mtime for p in checkpoints) if checkpoints else None
            try:
                event_dir = next(run.glob('runs/*'))
                events = EventAccumulator(str(event_dir), size_guidance={'scalars': 1}).Reload()
                telemetry['returns'] = {tag: events.Scalars(tag)[-1].value
                    for tag in ('charts/avg_episode_reward', 'charts/eval_return_det') if tag in events.Tags()['scalars']}
            except (StopIteration, OSError, ValueError):
                pass
            contract['telemetry'] = telemetry
        if shutil.disk_usage(ROOT).free < 30 * 1024**3 or any(c['telemetry']['co_runners'] for c in active.values()):
            write(OPS / 'ADMISSION_BLOCKED.json', dict(at=time.time(), reason='Disk reserve or GPU co-runner violation', active=active))
        slot_estimates = {}
        for slot, queue in CAMPAIGN['queue_by_gpu_slot'].items():
            seconds = 0.0
            available = True
            for job_id in queue:
                state = jobs[job_id]['state']
                if state in ('EVALUATION_FINISHED', 'FAILED', 'INTERRUPTED', 'BLOCKED', 'DEFERRED', 'TRAINING_FINISHED', 'EVALUATING'):
                    continue
                timing = OPS / 'timings' / (JOBS[job_id]['group'] + '.json')
                if not timing.exists():
                    available = False
                    break
                rate = read(timing)['measured_sps']
                step = active.get(job_id, {}).get('telemetry', {}).get('step', 4000)
                seconds += max(0, 200000 - max(4000, step)) / rate * 1.5
            slot_estimates[slot] = seconds if available else None
        write(OPS / 'campaign_live.json', dict(binding(), schema_version=3, observed_at=time.time(),
              remaining_training_seconds_by_slot=slot_estimates,
              eta_caveat='Pilot-based 50% margin. Excludes evaluation/audit/sync. Late cost may grow.',
              coordinator=identity(os.getpid()), active=active, jobs=jobs, gpu=sample,
              disk_free_bytes=__import__('shutil').disk_usage(ROOT).free))
        report()
        stop.wait(30)


def supervised_monitor(stop):
    try:
        monitor(stop)
    except BaseException:
        write(OPS / 'MONITOR_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))


def main():
    import torch
    install_signals()
    torch.set_num_threads(1)
    lease = (OPS / 'coordinator.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract_check()
    if (OPS / 'COORDINATOR_STARTED.json').exists():
        raise RuntimeError('Coordinator already attempted. Read-only recovery only, no retry/resume.')
    write(OPS / 'COORDINATOR_STARTED.json', dict(binding(), at=time.time(), identity=identity(os.getpid())))
    while True:
        latch_check()
        if (OPS / 'PREFLIGHT_FAILED.json').exists():
            raise RuntimeError('Preflight failed. Scientific queue blocked.')
        if (OPS / 'PREFLIGHT_DONE.json').exists():
            preflight_gate()
            break
        SHUTDOWN.wait(2)
    for run_id, job in JOBS.items():
        if (OUTPUT / run_id).exists() or (OPS / 'attempts' / run_id).exists():
            raise RuntimeError('Scientific run root reused')
        make_plan(job, OUTPUT / run_id, run_id)
    stop = threading.Event()
    thread = threading.Thread(target=supervised_monitor, args=(stop,), daemon=True)
    thread.start()
    try:
        # Wait for the first monitor publication rather than trusting a previous file.
        while not (OPS / 'campaign_live.json').exists():
            latch_check()
            if (OPS / 'MONITOR_FAILED.json').exists():
                raise RuntimeError('Initial monitor failed')
            SHUTDOWN.wait(1)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(train_slot, (0, 1, 2)))
        latch_check()
        selection = select_capsules()
        with LOCK:
            for row in selection:
                AUDITS[f'audit_a{row["seed"]}_{row["target_step"]}'] = row
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda seed: audit_seed(seed, selection), (1, 2, 3)))
    finally:
        stop.set()
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError('Monitor did not exit before final publication')
    report()
    all_ok = all(row['state'] == 'EVALUATION_FINISHED' for row in STATE.values()) and all(row['state'] == 'FINISHED' for row in AUDITS.values())
    write(OPS / 'CAMPAIGN_TERMINAL.json', dict(binding(), at=time.time(),
        state='COMPLETE' if all_ok else 'INCOMPLETE', local_sync='PENDING',
        runs={k: v['state'] for k, v in STATE.items()}, audits={k: v['state'] for k, v in AUDITS.items()}))
    final_hashes = {str(p.relative_to(ROOT)): digest(p) for p in
                    (ROOT / 'campaign_report.json', ROOT / 'campaign_report.md', OPS / 'CAMPAIGN_TERMINAL.json')}
    # Terminal and reports are immutable from here. Acknowledgment is separate.
    sync_gate(final_hashes=final_hashes)
    write(OPS / 'FINAL_SYNC_ACK.json', dict(binding(), at=time.time(), final_sha256=final_hashes,
          receipt=read(OPS / 'sync_receipt.json')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        with LOCK:
            for row in STATE.values():
                if row['state'] not in ('EVALUATION_FINISHED', 'FAILED', 'INTERRUPTED'):
                    row.update(state='BLOCKED', reason='Coordinator stopped or failed')
            for row in AUDITS.values():
                if row['state'] == 'PLANNED':
                    row.update(state='BLOCKED', reason='Coordinator stopped or failed')
        report()
        write(OPS / 'COORDINATOR_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
        raise
