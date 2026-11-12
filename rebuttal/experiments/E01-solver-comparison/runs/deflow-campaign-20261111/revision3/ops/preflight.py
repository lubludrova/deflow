"""Fresh evidence-bound strict CUDA preflight. No scientific runs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import platform
import re
import subprocess
import time
import traceback
from common import *

GROUPS = ['newton', 'anderson', 'anderson_highcap', 'anderson_tighter']


def one(group, slot, smoke):
    job = copy.deepcopy(next(j for j in contract_check()['jobs'] if j['group'] == group))
    job['argv'][job['argv'].index('--seed') + 1] = '0'
    job['gpu_slot'] = slot
    label = ('smoke_' if smoke else 'pilot_') + group
    run = ROOT / 'preflight' / label
    make_plan(job, run, label, steps=8000, smoke=smoke)
    receipt = execute(label, train_argv(job, run, steps=8000, smoke=smoke), slot, run)
    checkpoint = validate_run(run, 256 if smoke else 8000, 8 if smoke else 1024)
    if smoke:
        evaluate(label + '_eval', run, checkpoint, ROOT / 'preflight' / (label + '_eval'), 4, slot)
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(checkpoint.parent), size_guidance={'scalars': 0}).Reload()
        trained = [r for r in events.Scalars('loss/q_loss') if r.step >= 5000]
        if len(trained) < 2 or trained[-1].wall_time <= trained[0].wall_time:
            raise RuntimeError('Insufficient post-learning timing evidence')
        sps = (trained[-1].step - trained[0].step) / (trained[-1].wall_time - trained[0].wall_time)
        write(OPS / 'timings' / (group + '.json'), dict(group=group, measured_sps=sps,
            from_step=trained[0].step, to_step=trained[-1].step,
            pilot_seconds=receipt['elapsed_seconds'],
            preliminary_training_seconds_with_50pct_margin=196000 / sps * 1.5,
            caveat='Early strict post-learning pilot. Late cap500, evaluation and audit cost additional time.'))
    return run


def numerical(phase, fresh_bank=None):
    output = ROOT / 'preflight' / ('numerical_' + phase)
    argv = [PYTHON, str(ROOT / 'validation/numerical_preflight.py'), '--root', str(ROOT),
            '--output', str(output), '--phase', phase, '--device', 'cuda']
    if fresh_bank:
        argv += ['--fresh-bank', str(fresh_bank)]
    execute('numerical_' + phase, argv, 0)
    if read(output / 'verdict.json').get('pass') is not True:
        raise RuntimeError('Numerical preflight did not pass')


def fresh_bank(runs):
    import torch
    rows = []
    for group, run in runs.items():
        eligible = []
        for path in sorted(run.glob('capsule_*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            meta = payload.get('metadata', {})
            if (meta.get('role') == 'training/actor_grad' and meta.get('batch') == 1024
                    and len(payload.get('obs', [])) == 1024
                    and {'actor', 'q1', 'q2', 'loss_config'} <= payload.keys()):
                eligible.append((int(meta['global_step']), path))
        if not eligible:
            raise RuntimeError(f'Missing full-batch pilot capsule: {group}')
        step, path = max(eligible, key=lambda row: (row[0], str(row[1])))
        rows.append(dict(path=str(path), sha256=digest(path), actual_step=step, source=group,
                         rule='Latest eligible full-batch training/actor_grad capsule, lexical path tie break'))
    path = OPS / 'fresh_bank.json'
    write(path, rows)
    return path


def main():
    import torch
    install_signals()
    lease = (OPS / 'preflight.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((OPS / name).exists() for name in ('PREFLIGHT_DONE.json', 'PREFLIGHT_FAILED.json')):
        raise RuntimeError('Preflight already attempted in this immutable root')
    latch_check()
    contract_check()
    write(OPS / 'environment.json', dict(binding(), python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(), gpu_sample=gpu_sample(),
        disk_free_bytes=shutil.disk_usage(ROOT).free, package=str(PACKAGE), repo=str(REPO), output=str(OUTPUT)))
    if torch.__version__ != '2.5.1+cu124' or torch.version.cuda != '12.4' or not torch.cuda.is_available():
        raise RuntimeError('Pinned Torch/CUDA gate failed')
    lifecycle = subprocess.run([PYTHON, '-m', 'unittest', 'discover', '-s', str(OPS),
        '-p', 'test_lifecycle.py', '-v'], capture_output=True, text=True, timeout=120)
    (OPS / 'tests_lifecycle.log').write_text(lifecycle.stdout + lifecycle.stderr)
    if lifecycle.returncode:
        raise RuntimeError('Operational lifecycle tests failed')
    tests = ['deflow_rebuttal/test_solver.py', 'deflow_rebuttal/test_audit.py']
    for label, names in [('original', tests), ('precision', ['deflow_rebuttal/test_precision.py']),
                          ('numerical_gate', [str(ROOT / 'validation/test_numerical_preflight.py')])]:
        latch_check()
        result = subprocess.run([PYTHON, '-m', 'pytest', '-q', *names], cwd=PACKAGE,
            env=env(0), capture_output=True, text=True, timeout=1200)
        (OPS / ('tests_' + label + '.log')).write_text(result.stdout + result.stderr)
        if result.returncode or 'skipped' in result.stdout or not re.search(r'\b\d+ passed\b', result.stdout):
            raise RuntimeError(f'Required {label} tests failed')
        if label == 'original' and not re.search(r'\b16 passed\b', result.stdout):
            raise RuntimeError('Original sixteen tests not all retained')
    with (OPS / 'pip-freeze.txt').open('w') as stream:
        subprocess.run([PYTHON, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    numerical('fixed')
    def slot_stage(slot, groups, smoke):
        outputs = {}
        try:
            for group in groups:
                latch_check()
                outputs[group] = one(group, slot, smoke)
            return outputs
        except BaseException:
            write(OPS / 'PREFLIGHT_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
            raise
    runs = {}
    for smoke in (True, False):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(slot_stage, slot, groups, smoke) for slot, groups in
                [(0, ['newton']), (1, ['anderson', 'anderson_tighter']), (2, ['anderson_highcap'])]]
            for future in futures:
                values = future.result()
                if not smoke:
                    runs.update(values)
        latch_check()
    numerical('fresh', fresh_bank(runs))
    pilot_bytes = {group: sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
                   for group, run in runs.items()}
    projected = sum(pilot_bytes.values()) * (200000 / 8000) * 3
    write(OPS / 'disk_estimate.json', dict(pilot_bytes=pilot_bytes,
        projected_training_bytes=projected, remote_free_bytes=shutil.disk_usage(ROOT).free,
        caveat='Linear pilot extrapolation for12 runs only. Evaluation/audit artifacts and changing telemetry may add cost. No deletion assumed.'))
    if projected + 30 * 1024**3 > shutil.disk_usage(ROOT).free:
        raise RuntimeError('Projected training artifacts exceed remote reserve. Capacity review required.')
    evidence = {str((OPS / 'disk_estimate.json').relative_to(ROOT)): digest(OPS / 'disk_estimate.json')}
    for base in (ROOT / 'preflight', OPS / 'timings', OPS / 'attempts'):
        for path in base.rglob('*.json'):
            if path.name not in ('LIVE.json',):
                evidence[str(path.relative_to(ROOT))] = digest(path)
    for path in OPS.glob('tests_*.log'):
        evidence[str(path.relative_to(ROOT))] = digest(path)
    latch_check()
    write(OPS / 'PREFLIGHT_DONE.json', dict(binding(), finished_at=time.time(),
        counts={'smokes': 4, 'smoke_evaluations': 4, 'pilots': 4},
        evidence_sha256=evidence, scientific_runs_started=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(OPS / 'PREFLIGHT_FAILED.json', dict(at=time.time(), traceback=traceback.format_exc()))
        if not ACTIVE and not (OPS / 'CLEANUP_FAILED.json').exists():
            write(OPS / 'PREFLIGHT_TERMINAL_FAILED.json', dict(at=time.time(),
                state='FAILED', siblings_joined=True, owned_children_active=0))
        raise
