"""Approved operational overlay. Frozen scientific sources and live parents stay intact."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT / 'ops'))
import common as c
import campaign as original

STATE = {}
AUDITS = {}
ADOPTED = {}
CONFIG = {}
LOCAL = threading.local()
REPORT_LOCK = threading.RLock()
EXPAND = threading.Event()
MONITOR_STOP = threading.Event()
HISTORY = {}
WINDOW = []


def packing_check():
    contract = c.contract_check()
    config = c.read(BASE / 'manifest.json')
    if c.digest(BASE / 'manifest.json') != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Packing manifest anchor mismatch')
    if config['contract_sha256'] != os.environ['DEFLOW_CONTRACT_SHA256']:
        raise RuntimeError('Packing/scientific contract mismatch')
    for name, sha in config['files_sha256'].items():
        if c.digest(BASE / name) != sha:
            raise RuntimeError('Packing source mismatch: ' + name)
    flat = [job for queue in config['second_lanes'].values() for job in queue]
    adopted = list(config['adopted'])
    accepted = config['already_evaluated']
    if (len(flat + adopted + accepted) != 12
            or set(flat + adopted + accepted) != {j['id'] for j in contract['jobs']}
            or config['max_compute_per_gpu'] != 2 or config['retry_budget'] != 0
            or config['resume'] is not False):
        raise RuntimeError('Packing scope mismatch')
    jobs = {j['id']: j for j in contract['jobs']}
    for slot, queue in config['second_lanes'].items():
        if any(jobs[name]['gpu_slot'] != int(slot) for name in queue):
            raise RuntimeError('Seed/GPU assignment changed')
    return config


def known_running():
    with c.LOCK:
        rows = dict(ADOPTED, **c.ACTIVE)
    return {name: row for name, row in rows.items()
            if row.get('identity') and c.same_process(row['identity'])}


def check_occupancy(sample, running, slot=None, launching=False):
    mapping = c.read(c.OPS / 'execution_contract.json')['gpu_by_slot']
    for key, uuid in mapping.items():
        if slot is not None and int(key) != slot:
            continue
        known = {r['pid'] for r in running.values() if r['gpu_uuid'] == uuid}
        actual = {int(r[1]) for r in sample['processes'] if r[0] == uuid}
        if actual - known:
            raise RuntimeError('Unknown GPU process on slot ' + key)
        limit = 1 if launching else 2
        if len(known) > limit:
            raise RuntimeError('GPU concurrency bound on slot ' + key)
        gpu = next(r for r in sample['gpus'] if r[1] == uuid)
        if launching and int(gpu[3]) < 4096:
            raise RuntimeError('Less than 4 GiB free for new compute task')


def health_gate():
    packing_check()
    for path in (BASE / 'MONITOR_FAILED.json', BASE / 'ADMISSION_BLOCKED.json',
                 c.OPS / 'CLEANUP_FAILED.json', c.OPS / 'MONITOR_FAILED.json'):
        if path.exists():
            raise RuntimeError('Admission blocked: ' + str(path))
    live = c.read(BASE / 'campaign_live.json')
    if time.time() - live['observed_at'] > 120:
        raise RuntimeError('Packed monitor stale')
    if live['packing_sha256'] != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Packed monitor binding mismatch')
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Remote disk reserve')


def capacity(slot):
    packing_check()
    sample = c.gpu_sample()
    check_occupancy(sample, known_running(), slot, launching=True)
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Remote disk reserve')
    return sample


def execute(run_id, argv, slot, run=None, scientific=False, required=None):
    """Original attempt protocol, with two identity-checked lanes and serialized admission."""
    c.latch_check()
    packing_check()
    if scientific:
        c.preflight_gate()
        c.sync_gate(required)
    lane = getattr(LOCAL, 'lane', 0)
    lease = (BASE / f'gpu_{slot}_lane_{lane}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = c.OPS / 'attempts' / run_id
    process = owned = None
    claimed = False
    started = time.time()
    try:
        with (BASE / 'admission.lock').open('a') as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            c.latch_check()
            health_gate()
            observation = capacity(slot)
            attempt.mkdir(parents=True, exist_ok=False)
            claimed = True
            log = c.OPS / 'logs' / (run_id + '.log')
            record = dict(c.binding(), packing_sha256=os.environ['DEFLOW_PACKING_SHA256'],
                run_id=run_id, attempt_id=0, argv=argv, cwd=str(c.PACKAGE),
                gpu_slot=slot, gpu_uuid=c.env(slot)['CUDA_VISIBLE_DEVICES'], lane=lane,
                environment={k: c.env(slot)[k] for k in ('PYTHONPATH', 'DEFLOW_ENGINE', 'CUDA_VISIBLE_DEVICES')},
                command_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
                claimed_at=started, capacity_sample=observation, log=str(log),
                run_path=str(run) if run else None, retry_budget=0, resume=False,
                co_runners_at_claim={name: dict(identity=row['identity'],
                    progress=c.progress(Path(row['run_path']))) for name, row in known_running().items()
                    if row['gpu_slot'] == slot})
            c.write(attempt / 'CLAIM.json', record)
            packing_check()
            with log.open('x') as stream:
                process = subprocess.Popen(argv, cwd=c.PACKAGE, env=c.env(slot),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            owned = c.identity(process.pid)
            record.update(identity=owned, pid=process.pid, started_at=started, state='RUNNING')
            c.write(attempt / 'START.json', record)
            with c.LOCK:
                c.ACTIVE[run_id] = record
        while process.poll() is None:
            if c.stop_requested():
                c.write(attempt / 'STOP_RECEIPT.json', c.terminate_owned(process, owned))
                raise RuntimeError('Approved campaign STOP')
            c.write(attempt / 'LIVE.json', dict(record, observed_at=time.time(),
                telemetry=c.progress(run) if run else {}))
            c.SHUTDOWN.wait(2)
        code = process.wait()
        if c.owned_group_members(owned):
            c.write(attempt / 'STOP_RECEIPT.json', c.terminate_owned(process, owned))
            raise RuntimeError('Lingering descendants')
        terminal = dict(record, exit_code=code, finished_at=time.time(),
            elapsed_seconds=time.time() - started, state='EXITED')
        c.write(attempt / 'EXIT.json', terminal)
        if code:
            raise RuntimeError(f'{run_id} exited {code}: {log}')
        return terminal
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                c.write(attempt / 'STOP_RECEIPT.json', c.terminate_owned(process, owned))
            except BaseException:
                c.write(c.OPS / 'CLEANUP_FAILED.json', dict(run_id=run_id,
                    at=time.time(), traceback=traceback.format_exc()))
                raise
        if claimed:
            c.write(attempt / 'FAILED.json', dict(run_id=run_id, at=time.time(),
                exit_code=process.returncode if process else None, traceback=traceback.format_exc()))
        raise
    finally:
        with c.LOCK:
            c.ACTIVE.pop(run_id, None)
        lease.close()


def update(name, **values):
    with c.LOCK:
        STATE[name].update(values)


def adopted_training(name):
    """Keep the original Popen parent alive to observe the real exit code."""
    attempt = c.OPS / 'attempts' / name
    original_identity = CONFIG['adopted'][name]['identity']
    while True:
        c.latch_check()
        if (attempt / 'FAILED.json').exists():
            raise RuntimeError('Original scientific attempt failed: ' + name)
        if (attempt / 'EXIT.json').exists():
            terminal = c.read(attempt / 'EXIT.json')
            if terminal['identity'] != original_identity or terminal['exit_code'] != 0:
                raise RuntimeError('Original attempt terminal mismatch: ' + name)
            if (attempt / 'ACCEPTED.json').exists():
                accepted = c.read(attempt / 'ACCEPTED.json')
                if not all(c.digest(ROOT / p) == sha for p, sha in accepted['artifacts_sha256'].items()):
                    raise RuntimeError('Original accepted artifacts mismatch')
                return terminal
            failure = c.OPS / 'failures' / (name + '.json')
            if failure.exists():
                raise RuntimeError('Original post-training validation failed: ' + name)
        elif not c.same_process(original_identity):
            # The parent may be publishing EXIT. Never infer success from absence.
            c.SHUTDOWN.wait(5)
            if not (attempt / 'EXIT.json').exists():
                raise RuntimeError('Original process missing without terminal evidence: ' + name)
        c.SHUTDOWN.wait(2)


def train_and_evaluate(name, adopted=False):
    job = original.JOBS[name]
    run = c.OUTPUT / name
    if adopted:
        training = adopted_training(name)
    else:
        c.sync_gate()
        update(name, state='RUNNING', started_at=time.time())
        training = execute(name, c.train_argv(job, run), job['gpu_slot'], run, scientific=True)
    checkpoint = c.validate_run(run, 200000, 1024)
    if not adopted:
        c.accepted(name, [checkpoint, run / 'DONE.json', run / 'manifest.json', run / 'runtime_precision.json'])
    update(name, state='TRAINING_FINISHED', training=training, checkpoint=str(checkpoint))
    with c.LOCK:
        ADOPTED.pop(name, None)
    update(name, state='EVALUATING')
    evaluation = c.evaluate(name + '_eval', run, checkpoint, ROOT / 'evaluations' / name,
        1000, job['gpu_slot'], scientific=True)
    metrics = c.read(ROOT / 'evaluations' / name / 'evaluation.json')['summary']
    update(name, state='EVALUATION_FINISHED', evaluation=evaluation, metrics=metrics)
    c.sync_gate(name + '_eval')
    report()


def lane_worker(slot, lane, queue):
    LOCAL.lane = lane
    if lane == 1 and slot != 0:
        while not EXPAND.wait(2):
            c.latch_check()
            if (BASE / 'CANARY_BLOCKED.json').exists():
                for name in queue:
                    update(name, state='BLOCKED', reason='Co-location canary did not pass')
                return
    for index, name in enumerate(queue):
        try:
            train_and_evaluate(name, adopted=lane == 0)
        except BaseException:
            error = traceback.format_exc()
            update(name, state='FAILED', error=error)
            c.write(BASE / 'failures' / (name + '.json'), dict(run_id=name,
                at=time.time(), traceback=error))
            for pending in queue[index + 1:]:
                update(pending, state='BLOCKED', reason='Lane blocked by ' + name)
            if name == 'anderson_s1' and not EXPAND.is_set():
                c.write(BASE / 'CANARY_BLOCKED.json', dict(at=time.time(), reason=error))
            report()
            return
        if lane == 1 and slot == 0 and not EXPAND.is_set():
            while not EXPAND.wait(2):
                c.latch_check()
                if (BASE / 'CANARY_BLOCKED.json').exists():
                    for pending in queue[index + 1:]:
                        update(pending, state='BLOCKED', reason='Co-location canary did not pass')
                    return


def report():
    with REPORT_LOCK:
        with c.LOCK:
            rows = json.loads(json.dumps(STATE))
            audits = json.loads(json.dumps(AUDITS))
        value = dict(c.binding(), packing_sha256=os.environ['DEFLOW_PACKING_SHA256'],
            observed_at=time.time(), runs=rows, audits=audits,
            counts=dict(runs=12, evaluated=sum(r['state'] == 'EVALUATION_FINISHED' for r in rows.values()),
                audits=9, audits_finished=sum(r['state'] == 'FINISHED' for r in audits.values())),
            original_controller_reports='campaign_report.json and ops/campaign_live.json',
            note='Original coordinator scheduling errors after handover are separate from scientific attempt status.')
        c.write(BASE / 'campaign_report.json', value)
        lines = ['# DEFlow packed campaign', '', '| Run | State | U4 | Success | Return |',
                 '| --- | --- | ---: | ---: | ---: |']
        for name, row in rows.items():
            m = row.get('metrics', {})
            lines.append('| ' + ' | '.join(str(x) for x in
                [name, row['state'], m.get('u4', ''), m.get('total_success', ''), m.get('mean_return', '')]) + ' |')
        lines.extend(['', '| Audit | State |', '| --- | --- |'])
        lines.extend('| ' + name + ' | ' + row['state'] + ' |' for name, row in audits.items())
        lines.extend(['', 'Shared-GPU and exclusive intervals must be analyzed separately.',
                      'Raw telemetry and audit raw.pt remain on the server. All failures and invalid references are retained.'])
        (BASE / 'campaign_report.md').write_text('\n'.join(lines) + '\n')


def rate(points):
    if len(points) < 2 or points[-1][0] - points[0][0] < 60:
        return None
    return (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])


def canary_score(rates, baselines):
    return sum(rates[name] / baselines[name] for name in baselines)


def monitor():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    canary_start = None
    canary_points = {}
    while not MONITOR_STOP.is_set():
        now = time.time()
        packing_check()
        sample = c.gpu_sample()
        WINDOW.append(sample)
        WINDOW[:] = [s for s in WINDOW if now - s['at'] <= 600]
        running = known_running()
        try:
            check_occupancy(sample, running)
            if shutil.disk_usage(ROOT).free < 30 * 1024**3:
                raise RuntimeError('Remote disk reserve')
        except RuntimeError:
            c.write(BASE / 'ADMISSION_BLOCKED.json', dict(at=now, traceback=traceback.format_exc()))
        active = {}
        for name, record in running.items():
            run = Path(record['run_path'])
            telemetry = c.progress(run)
            points = HISTORY.setdefault(name, [])
            points.append((now, telemetry['step']))
            points[:] = [p for p in points if now - p[0] <= 600]
            speed = rate(points)
            telemetry.update(delta_sps=speed,
                eta_seconds=(200000 - telemetry['step']) / speed if speed and name in STATE else None)
            gpu_rows = [r for s in WINDOW for r in s['gpus'] if r[1] == record['gpu_uuid']]
            utilization = sorted(int(r[4]) for r in gpu_rows)
            telemetry['gpu_utilization_window'] = dict(samples=len(utilization),
                mean=sum(utilization) / len(utilization), p95=utilization[int(.95 * (len(utilization) - 1))],
                peak=max(utilization), memory_used_mib=int(gpu_rows[-1][2]))
            telemetry['co_runners'] = [dict(run_id=k, pid=v['pid']) for k, v in running.items()
                if k != name and v['gpu_uuid'] == record['gpu_uuid']]
            checkpoints = list(run.glob('runs/*/*.pt'))
            telemetry['checkpoint_age_seconds'] = now - max(p.stat().st_mtime for p in checkpoints) if checkpoints else None
            try:
                event_dir = next(run.glob('runs/*'))
                events = EventAccumulator(str(event_dir), size_guidance={'scalars': 1}).Reload()
                telemetry['scalars'] = {tag: dict(step=events.Scalars(tag)[-1].step,
                    value=events.Scalars(tag)[-1].value, wall_time=events.Scalars(tag)[-1].wall_time)
                    for tag in ('charts/avg_episode_reward', 'charts/eval_return_det') if tag in events.Tags()['scalars']}
                telemetry['returns'] = {k: v['value'] for k, v in telemetry['scalars'].items()}
            except (StopIteration, OSError, ValueError):
                pass
            if name in STATE and telemetry['last_telemetry_age_seconds'] is not None and telemetry['last_telemetry_age_seconds'] > 3600:
                c.write(BASE / ('STALE_' + name + '.json'), dict(at=now, telemetry=telemetry))
            active[name] = dict(record, telemetry=telemetry)
        with c.LOCK:
            jobs_snapshot = json.loads(json.dumps(STATE))
        c.write(BASE / 'campaign_live.json', dict(c.binding(),
            packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], observed_at=now,
            coordinator=c.identity(os.getpid()), active=active, jobs=jobs_snapshot, gpu=sample,
            disk_free_bytes=shutil.disk_usage(ROOT).free, eta_caveat='Recent ten-minute per-job window. Shared GPU interference and late cost may change.'))
        with (BASE / 'resource_history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=now, gpu=sample, active={k: v['telemetry'] for k, v in active.items()})) + '\n')
        if not EXPAND.is_set() and not (BASE / 'CANARY_BLOCKED.json').exists():
            names = ('newton_s1', 'anderson_s1')
            ready = all(n in active for n in names) and active.get('anderson_s1', {}).get('telemetry', {}).get('step', 0) >= CONFIG['canary_min_step']
            if ready:
                if canary_start is None:
                    canary_start = now
                for name in names:
                    canary_points.setdefault(name, []).append((now, active[name]['telemetry']['step']))
                if now - canary_start >= CONFIG['canary_seconds']:
                    rates = {n: rate(canary_points[n]) for n in names}
                    progressing = all(v is not None and v > 0 for v in rates.values())
                    score = canary_score(rates, CONFIG['canary_solo_sps']) if progressing else 0
                    passed = progressing and score >= CONFIG['minimum_normalized_progress']
                    if any(active[n]['telemetry'].get('last_solver_call', {}).get('common_abs_residual', {}).get('nonfinite', 0) for n in names):
                        passed = False
                    if (BASE / 'ADMISSION_BLOCKED.json').exists():
                        passed = False
                    receipt = dict(at=now, seconds=now-canary_start, rates=rates, normalized_progress=score,
                        threshold=CONFIG['minimum_normalized_progress'], solo_baselines=CONFIG['canary_solo_sps'],
                        caveat='Operational progress check across evolving policies, not a controlled speedup experiment.')
                    c.write(BASE / ('CANARY_PASSED.json' if passed else 'CANARY_BLOCKED.json'), receipt)
                    if passed:
                        EXPAND.set()
        report()
        MONITOR_STOP.wait(30)


def supervised_monitor():
    try:
        monitor()
    except BaseException:
        c.write(BASE / 'MONITOR_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
        c.write(BASE / 'CANARY_BLOCKED.json', dict(at=time.time(), reason='Monitor failed'))


def select_capsules():
    # Same frozen selection implementation, writing to the operational overlay.
    import torch
    selection = []
    for seed in (1, 2, 3):
        name = f'anderson_s{seed}'
        candidates = []
        for path in sorted((c.OUTPUT / name).glob('capsule_*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            m = payload.get('metadata', {})
            if (m.get('role') == 'training/actor_grad' and m.get('batch') == 1024
                    and m.get('global_step') is not None and len(payload.get('obs', [])) == 1024
                    and {'actor', 'q1', 'q2', 'loss_config'} <= payload.keys()):
                candidates.append((int(m['global_step']), path))
        for target in (25000, 100000, 200000):
            row = dict(seed=seed, target_step=target, run_id=name,
                source_run_state=STATE[name]['state'], rule='Nearest eligible full-batch training/actor_grad capsule. Ties: smaller actual step, then lexical path.')
            if candidates:
                step, path = min(candidates, key=lambda pair: (abs(pair[0]-target), pair[0], str(pair[1])))
                row.update(actual_step=step, path=str(path), sha256=c.digest(path), state='PLANNED')
            else:
                row.update(state='MISSING', reason='No eligible capsule. No substitute checkpoint used.')
            selection.append(row)
    c.write(BASE / 'audit_selection.json', selection)
    return selection


def main():
    global CONFIG
    CONFIG = packing_check()
    c.preflight_gate()
    if '--check' in sys.argv:
        live = c.read(c.OPS / 'campaign_live.json')
        if not c.same_process(live['coordinator']):
            raise RuntimeError('Original coordinator not alive')
        running = {}
        for name, expected in CONFIG['adopted'].items():
            actual = c.read(c.OPS / 'attempts' / name / 'START.json')
            if actual != expected or not c.same_process(actual['identity']):
                raise RuntimeError('Adopted attempt changed: ' + name)
            running[name] = actual
        check_occupancy(c.gpu_sample(), running)
        for queue in CONFIG['second_lanes'].values():
            for name in queue:
                if (c.OPS / 'attempts' / name).exists():
                    raise RuntimeError('Pending attempt already claimed: ' + name)
        print(json.dumps(dict(result='ok', frozen_sources_verified=True,
            packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], adopted=list(running))))
        return
    import torch
    torch.set_num_threads(1)
    c.install_signals()
    c.sync_gate('packing_v1_configuration')
    lease = (BASE / 'coordinator.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = BASE / 'COORDINATOR_STARTED.json'
    with started.open('x') as stream:
        json.dump(dict(at=time.time(), identity=c.identity(os.getpid()),
            packing_sha256=os.environ['DEFLOW_PACKING_SHA256']), stream, indent=2)
    snapshot = c.read(c.ROOT / 'campaign_report.json')
    for name, job in original.JOBS.items():
        STATE[name] = dict(state='PLANNED', seed=job['seed'], gpu_slot=job['gpu_slot'])
    for name in CONFIG['already_evaluated']:
        if snapshot['runs'][name]['state'] != 'EVALUATION_FINISHED':
            raise RuntimeError('Previously accepted evaluation changed')
        STATE[name] = snapshot['runs'][name]
    for name, row in CONFIG['adopted'].items():
        if not c.same_process(row['identity']):
            raise RuntimeError('Adopted process vanished before handover')
        ADOPTED[name] = row
        update(name, state='RUNNING', started_at=row['started_at'], original_parent_retained=True)
    AUDITS.update(json.loads(json.dumps(original.AUDITS)))
    c.health_gate = health_gate
    c.execute = execute
    original.STATE = STATE
    original.AUDITS = AUDITS
    original.execute = execute
    original.report = report
    # A durable administrative latch blocks only future old-controller launches.
    # Its active execute loops do not consult this marker and retain real exits.
    c.write(BASE / 'HANDOVER.json', dict(c.binding(), at=time.time(),
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], adopted=ADOPTED,
        old_parent_retained=True, reason='User approved two training processes per GPU.'))
    c.write(c.OPS / 'ADMISSION_BLOCKED.json', dict(at=time.time(),
        reason='Administrative handover to ops/packing-v1. Current children and original exit observers remain alive.',
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256']))
    thread = threading.Thread(target=supervised_monitor, daemon=True)
    thread.start()
    while not (BASE / 'campaign_live.json').exists():
        if (BASE / 'MONITOR_FAILED.json').exists():
            raise RuntimeError('Initial packed monitor failed')
        c.SHUTDOWN.wait(1)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(lane_worker, slot, 0, [f'newton_s{slot+1}']) for slot in range(3)]
            futures += [pool.submit(lane_worker, slot, 1, CONFIG['second_lanes'][str(slot)]) for slot in range(3)]
            for future in futures:
                future.result()
        selection = select_capsules()
        for row in selection:
            AUDITS[f'audit_a{row["seed"]}_{row["target_step"]}'] = row
        def audits(seed):
            LOCAL.lane = 0
            original.audit_seed(seed, selection)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(audits, (1, 2, 3)))
    finally:
        MONITOR_STOP.set()
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError('Packed monitor failed to quiesce')
    report()
    complete = all(r['state'] == 'EVALUATION_FINISHED' for r in STATE.values()) and all(r['state'] == 'FINISHED' for r in AUDITS.values())
    c.write(BASE / 'CAMPAIGN_TERMINAL.json', dict(c.binding(), at=time.time(),
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], state='COMPLETE' if complete else 'INCOMPLETE',
        runs={k: v['state'] for k, v in STATE.items()}, audits={k: v['state'] for k, v in AUDITS.items()}))
    paths = [BASE / name for name in ('campaign_report.json', 'campaign_report.md', 'CAMPAIGN_TERMINAL.json')]
    c.accepted('packing_v1_final', paths)
    hashes = {str(p.relative_to(ROOT)): c.digest(p) for p in paths}
    c.sync_gate(final_hashes=hashes)
    c.write(BASE / 'FINAL_SYNC_ACK.json', dict(c.binding(), at=time.time(),
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], final_sha256=hashes,
        receipt=c.read(c.OPS / 'sync_receipt.json')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if '--check' not in sys.argv:
            c.write(BASE / 'COORDINATOR_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
        raise
