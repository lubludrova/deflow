"""Run the nine approved paired audits on idle GPU 1, retaining live parents."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
sys.path.insert(0, str(ROOT / 'ops'))
import common as c


def packing_check():
    contract = c.contract_check()
    config = c.read(BASE / 'manifest.json')
    if c.digest(BASE / 'manifest.json') != os.environ['DEFLOW_PACKING_SHA256']:
        raise RuntimeError('Idle-audit manifest anchor mismatch')
    if config['contract_sha256'] != os.environ['DEFLOW_CONTRACT_SHA256']:
        raise RuntimeError('Scientific contract changed')
    for relative, sha in config['files_sha256'].items():
        if c.digest(BASE / relative) != sha:
            raise RuntimeError('Idle-audit source changed: ' + relative)
    for relative, sha in config['dependencies_sha256'].items():
        if c.digest(ROOT / relative) != sha:
            raise RuntimeError('Pinned operational dependency changed: ' + relative)
    expected = {job['id'] for job in contract['jobs']}
    if set(config['snapshot']['runs']) != expected or len(expected) != 12:
        raise RuntimeError('Training scope changed')
    audit_ids = {f'audit_a{s}_{t}' for s in (1, 2, 3) for t in (25000, 100000, 200000)}
    if (len(config['audit_order']) != 9 or set(config['audit_order']) != audit_ids
            or config['audit_gpu_slot'] != 1 or config['retry_budget'] != 0
            or config['resume'] is not False):
        raise RuntimeError('Audit scope or retry policy changed')
    return config


def load_packed(config):
    path = ROOT / 'ops/packing-v1/controller.py'
    spec = importlib.util.spec_from_file_location('pinned_packing_library', path)
    packed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packed)
    packed.BASE = BASE
    packed.CONFIG = config
    packed.packing_check = packing_check
    packed.EXPAND.set()
    return packed


def choose_snapshot(config, block_old_admission=False):
    """Let an already-started old evaluation finish before handing over admission."""
    previous = ROOT / 'ops' / config['previous_overlay']
    while True:
        c.latch_check()
        with (previous / 'admission.lock').open('a') as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            snapshot = c.read(previous / 'campaign_report.json')
            live = c.read(previous / 'campaign_live.json')
            if not c.same_process(live['coordinator']):
                raise RuntimeError('Previous parent coordinator is missing')
            busy = any(row['state'] not in ('RUNNING', 'FAILED', 'EVALUATION_FINISHED')
                       for row in snapshot['runs'].values())
            busy = busy or time.time()-live['observed_at'] > 120
            active = live['active']
            if not busy and all(name in snapshot['runs'] and snapshot['runs'][name]['state'] == 'RUNNING'
                                and c.same_process(row['identity']) for name, row in active.items()):
                running = {name for name, row in snapshot['runs'].items() if row['state'] == 'RUNNING'}
                if running == set(active):
                    if block_old_admission:
                        c.write(previous / 'ADMISSION_BLOCKED.json', dict(at=time.time(),
                            reason='Administrative handover to idle-audits-v1. Retain existing children and exit observers.',
                            packing_sha256=os.environ['DEFLOW_PACKING_SHA256']))
                    return snapshot, active, live
        c.SHUTDOWN.wait(2)


def validate_audit(summary, raw_path, capsule):
    if (summary.get('batch') != 1024 or summary.get('loss_kind') != 'configured_SAC_actor_loss'
            or summary.get('precision_policy') != 'strict_fp32'
            or summary.get('input_kind') != 'campaign_same_capsule'
            or summary.get('runtime_precision', {}).get('matmul_allow_tf32') is not False
            or summary.get('anderson_sweep_caps') != [25, 100, 500]):
        raise RuntimeError('Paired audit scientific protocol mismatch')
    hashes = summary['hashes']
    if hashes.get('checkpoint') != capsule['sha256'] or hashes.get('inputs') != capsule['sha256']:
        raise RuntimeError('Audit did not use the selected identical capsule')
    if summary['reference_valid'] + summary['reference_invalid'] != 1024:
        raise RuntimeError('Reference denominator mismatch')
    if summary['reference_invalid']:
        for label in ('anderson_cap25', 'anderson_cap100', 'anderson_cap500', 'newton'):
            row = summary[label]
            if row.get('reference_gradient_available') or row.get('gradient_relative_difference') is not None:
                raise RuntimeError('Invalid full-batch reference used for a gradient comparison')
    if not raw_path.is_file() or not raw_path.stat().st_size:
        raise RuntimeError('Missing audit raw artifact')


def audit_worker(packed, selection):
    packed.LOCAL.lane = 0
    by_id = {f'audit_a{row["seed"]}_{row["target_step"]}': row for row in selection}
    blocked = set()
    for name in packed.CONFIG['audit_order']:
        row = by_id[name]
        if row['seed'] in blocked:
            packed.AUDITS[name].update(state='BLOCKED', reason='Earlier audit for this seed failed')
            continue
        try:
            if row['state'] == 'MISSING':
                packed.AUDITS[name] = row
                blocked.add(row['seed'])
                continue
            c.sync_gate('idle_audits_selection')
            packing_check()
            sample = c.gpu_sample()
            uuid = c.contract_check()['gpu_by_slot']['1']
            if any(proc[0] == uuid for proc in sample['processes']):
                raise RuntimeError('Audit GPU must be completely exclusive')
            path = Path(row['path'])
            if c.digest(path) != row['sha256']:
                raise RuntimeError('Selected capsule changed')
            output = ROOT / 'audits' / name
            with c.LOCK:
                packed.AUDITS[name] = dict(row, state='RUNNING', gpu_slot=1, gpu_uuid=uuid,
                    started_at=time.time())
            packed.report()
            result = packed.execute(name, [c.PYTHON, '-m', 'deflow_rebuttal.audit',
                '--engine', str(c.REPO / 'src/docking_full/sac_anderson_flow.py'),
                '--checkpoint', str(path), '--inputs', str(path), '--anderson-caps', '25', '100', '500',
                '--anderson-tol', '1e-5', '--output', str(output), '--device', 'cuda',
                '--warmup', '2', '--repeats', '10', '--precision-policy', 'strict_fp32',
                '--require-same-capsule'], 1, output, scientific=True)
            summary = c.read(output / 'summary.json')
            validate_audit(summary, output / 'raw.pt', row)
            # The raw masks and gradients are small enough to preserve in the verified compact mirror.
            c.accepted(name, [output / 'summary.json', output / 'raw.pt', path])
            with c.LOCK:
                packed.AUDITS[name] = dict(row, state='FINISHED', gpu_slot=1, gpu_uuid=uuid,
                    receipt=result, summary=summary)
            packed.report()
            c.sync_gate(name)
        except BaseException:
            error = traceback.format_exc()
            with c.LOCK:
                packed.AUDITS[name] = dict(row, state='FAILED', traceback=error)
            c.write(BASE / 'failures' / (name + '.json'), packed.AUDITS[name])
            blocked.add(row['seed'])
            packed.report()
    packed.report()
    try:
        import report
        outputs = report.build_tables(ROOT, BASE)
        c.accepted('idle_audit_tables', outputs)
        c.sync_gate('idle_audit_tables')
    except BaseException:
        c.write(BASE / 'ANALYSIS_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))


def main():
    config = packing_check()
    c.preflight_gate()
    packed = load_packed(config)
    snapshot, active, old_live = choose_snapshot(config)
    uuid = c.contract_check()['gpu_by_slot']['1']
    sample = c.gpu_sample()
    if any(proc[0] == uuid for proc in sample['processes']):
        raise RuntimeError('GPU 1 is not idle')
    for name in config['audit_order']:
        if (c.OPS / 'attempts' / name).exists() or (ROOT / 'audits' / name).exists():
            raise RuntimeError('Audit already attempted: ' + name)
    if '--check' in sys.argv:
        print(json.dumps(dict(result='ok', adopted=list(active), idle_gpu_uuid=uuid,
            original_supervisor_verified=c.same_process(old_live['coordinator']))))
        return
    if '--prepare' in sys.argv:
        packed.STATE.update(snapshot['runs'])
        selection = packed.select_capsules()
        if len(selection) != 9 or any(row['state'] == 'MISSING' for row in selection):
            raise RuntimeError('All nine prescribed capsule selections must be available before handover')
        capsule_paths = list({row['path'] for row in selection})
        c.accepted('idle_audits_selection', [BASE / 'audit_selection.json', *map(Path, capsule_paths)])
        print(json.dumps(dict(result='prepared', selection=selection)))
        return
    import torch
    torch.set_num_threads(1)
    c.install_signals()
    lease = (BASE / 'coordinator.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (BASE / 'COORDINATOR_STARTED.json').open('x') as stream:
        json.dump(dict(at=time.time(), identity=c.identity(os.getpid()),
            packing_sha256=os.environ['DEFLOW_PACKING_SHA256']), stream, indent=2)
    snapshot, active, old_live = choose_snapshot(config, block_old_admission=True)
    packed.STATE.update(snapshot['runs'])
    packed.ADOPTED.update(active)
    packed.AUDITS.update({name: dict(state='PLANNED', seed=int(name.split('_')[1][1:]),
        target_step=int(name.split('_')[2])) for name in config['audit_order']})
    c.health_gate = packed.health_gate
    c.execute = packed.execute
    thread = threading.Thread(target=packed.supervised_monitor, daemon=True)
    thread.start()
    while not (BASE / 'campaign_live.json').exists():
        if (BASE / 'MONITOR_FAILED.json').exists():
            raise RuntimeError('Initial idle-audit monitor failed')
        c.SHUTDOWN.wait(1)
    c.sync_gate('idle_audits_configuration')
    c.write(BASE / 'HANDOVER.json', dict(c.binding(), at=time.time(),
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], previous_coordinator=old_live['coordinator'],
        retained_parents=active, snapshot=snapshot,
        reason='User authorized all nine prescribed audits on the idle GPU while training continues.'))
    selection = c.read(BASE / 'audit_selection.json')
    for row in selection:
        packed.AUDITS[f'audit_a{row["seed"]}_{row["target_step"]}'] = row
    try:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = []
            for name, record in active.items():
                def training(job=name, observed=record):
                    packed.LOCAL.lane = observed.get('lane', 0)
                    try:
                        packed.train_and_evaluate(job, adopted=True)
                    except BaseException:
                        packed.update(job, state='FAILED', error=traceback.format_exc())
                        c.write(BASE / 'failures' / (job + '.json'), dict(run_id=job,
                            at=time.time(), traceback=traceback.format_exc()))
                        packed.report()
                futures.append(pool.submit(training))
            futures.append(pool.submit(audit_worker, packed, selection))
            for future in futures:
                future.result()
    finally:
        packed.MONITOR_STOP.set()
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError('Monitor did not quiesce before final report')
    packed.report()
    complete = all(r['state'] == 'EVALUATION_FINISHED' for r in packed.STATE.values()) and all(r['state'] == 'FINISHED' for r in packed.AUDITS.values())
    c.write(BASE / 'CAMPAIGN_TERMINAL.json', dict(c.binding(), at=time.time(),
        state='COMPLETE' if complete else 'INCOMPLETE', packing_sha256=os.environ['DEFLOW_PACKING_SHA256'],
        runs={k: v['state'] for k, v in packed.STATE.items()}, audits={k: v['state'] for k, v in packed.AUDITS.items()}))
    paths = [BASE / name for name in ('campaign_report.json', 'campaign_report.md', 'CAMPAIGN_TERMINAL.json')]
    c.accepted('idle_audits_final', paths)
    hashes = {str(path.relative_to(ROOT)): c.digest(path) for path in paths}
    c.sync_gate(final_hashes=hashes)
    c.write(BASE / 'FINAL_SYNC_ACK.json', dict(c.binding(), at=time.time(),
        packing_sha256=os.environ['DEFLOW_PACKING_SHA256'], final_sha256=hashes,
        receipt=c.read(c.OPS / 'sync_receipt.json')))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if '--check' not in sys.argv and '--prepare' not in sys.argv:
            c.write(BASE / 'COORDINATOR_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
        raise
