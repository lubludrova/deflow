"""Small fail-closed primitives for the frozen strict-FP32 campaign."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / 'ops'
PACKAGE = ROOT / 'package'
REPO = ROOT / 'DEFlow'
OUTPUT = ROOT / 'results'
PYTHON = str(ROOT / 'venv/bin/python')
LOCK = threading.RLock()
ACTIVE = {}
SHUTDOWN = threading.Event()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contract_check():
    anchor = os.environ.get('DEFLOW_CONTRACT_SHA256')
    path = OPS / 'execution_contract.json'
    if not anchor or digest(path) != anchor:
        raise RuntimeError('Missing or mismatched external contract SHA256 anchor')
    contract = read(path)
    if contract['schema_version'] != 3 or contract['precision_policy'] != 'strict_fp32':
        raise RuntimeError('Wrong campaign schema or precision policy')
    required_files = {'RUNNER_PLAN.md', 'ops/common.py', 'ops/preflight.py', 'ops/campaign.py',
        'ops/sync_index.py', 'ops/sync_local.py', 'package/night_campaign.json',
        'package/engine_sha256.json', 'package/VALIDATION_CONVERGENCE.json',
        'package/deflow_rebuttal/precision.py', 'package/REVIEWED_SOURCE_MAP.json',
        'validation/numerical_preflight.py',
        'validation/historical_fixed_bank.json'}
    if not required_files <= contract['frozen_files'].keys():
        raise RuntimeError('Incomplete frozen source inventory')
    for base in (PACKAGE / 'deflow_rebuttal', REPO / 'src/docking_full', ROOT / 'validation', OPS):
        for source in base.glob('*.py'):
            if str(source.relative_to(ROOT)) not in contract['frozen_files']:
                raise RuntimeError(f'Unpinned runtime source: {source}')
    for name, expected in contract['frozen_files'].items():
        target = ROOT / name
        if target.resolve().is_relative_to(ROOT.resolve()) is False or digest(target) != expected:
            raise RuntimeError(f'Frozen source mismatch: {name}')
    campaign = read(PACKAGE / 'night_campaign.json')
    if contract['jobs'] != campaign['jobs'] or contract['queue_by_gpu_slot'] != campaign['queue_by_gpu_slot']:
        raise RuntimeError('Contract jobs differ from immutable original campaign')
    if len(contract['jobs']) != 12 or contract.get('retry_budget') != 0 or contract.get('resume') is not False:
        raise RuntimeError('Scientific scope mismatch')
    return contract


def binding():
    contract = contract_check()
    return {'campaign_id': contract['campaign_id'],
            'contract_sha256': os.environ['DEFLOW_CONTRACT_SHA256'],
            'precision_policy': 'strict_fp32'}


def check_sources():
    contract_check()


def stop_requested():
    return SHUTDOWN.is_set() or (OPS / 'STOP.json').exists()


def latch_check(wait_pause=True):
    while True:
        if stop_requested():
            raise RuntimeError('Campaign STOP requested')
        if (OPS / 'PREFLIGHT_FAILED.json').exists():
            raise RuntimeError('Preflight failure blocks new work')
        if not (OPS / 'PAUSE.json').exists():
            return
        if not wait_pause:
            raise RuntimeError('Campaign PAUSE requested')
        SHUTDOWN.wait(1)


def install_signals():
    def handler(signum, frame):
        SHUTDOWN.set()
        write(OPS / 'STOP.json', dict(at=time.time(), signal=signum, reason='Supervisor signal'))
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def env(slot):
    contract = contract_check()
    return dict(os.environ, PYTHONPATH=str(PACKAGE),
                DEFLOW_ENGINE=str(REPO / 'src/docking_full/sac_anderson_flow.py'),
                CUDA_VISIBLE_DEVICES=contract['gpu_by_slot'][str(slot)], PYTHONUNBUFFERED='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def gpu_sample():
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    return {'at': time.time(), 'gpus': [r.split(', ') for r in rows.strip().splitlines()],
            'processes': [r.split(', ') for r in processes.strip().splitlines() if r]}


def capacity(slot):
    sample = gpu_sample()
    uuid = contract_check()['gpu_by_slot'][str(slot)]
    row = next(r for r in sample['gpus'] if r[1] == uuid)
    if any(r[0] == uuid for r in sample['processes']):
        raise RuntimeError(f'Assigned GPU is occupied: slot {slot}')
    if int(row[3]) < 4096 or int(row[4]) > 10 or int(row[2]) > 256:
        raise RuntimeError(f'GPU unavailable: {row}')
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError('Remote disk below 30 GiB reserve')
    return sample


def identity(pid):
    base = Path(f'/proc/{pid}')
    stat = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': pid, 'process_start_ticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'process_group': int(stat[2]), 'session': int(stat[3]),
            'cmdline': (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace'),
            'cwd': str((base / 'cwd').resolve())}


def same_process(expected):
    try:
        current = identity(expected['pid'])
        return all(current[k] == expected[k] for k in ('pid', 'process_start_ticks', 'boot_id', 'process_group', 'session'))
    except (OSError, ValueError):
        return False


def owned_group_members(expected):
    if expected is None:
        return []
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(') ', 1)[1].split()
            if fields[0] == 'Z' or int(fields[2]) != expected['process_group']:
                continue
            member = identity(int(path.parent.name))
            if (member['boot_id'] != expected['boot_id'] or member['session'] != expected['session']
                    or member['process_start_ticks'] < expected['process_start_ticks']):
                raise RuntimeError('Owned process-group identity conflict')
            members.append(member)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def terminate_owned(process, expected):
    receipt = {'at': time.time(), 'identity': expected, 'signals': []}
    if expected is None or expected['process_group'] != process.pid:
        if process.poll() is not None:
            return dict(receipt, already_exited=process.returncode)
        raise RuntimeError('Refusing termination without matching owned process-group identity')
    members = owned_group_members(expected)
    if process.poll() is None and not same_process(expected):
        raise RuntimeError('Refusing termination with changed leader identity')
    receipt['members_before'] = members
    if not members:
        return dict(receipt, already_exited=process.poll())
    for signum, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        members = owned_group_members(expected)
        if not members:
            break
        os.killpg(process.pid, signum)
        receipt['signals'].append(signal.Signals(signum).name)
        if signum == signal.SIGTERM:
            os.killpg(process.pid, signal.SIGCONT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            if not owned_group_members(expected):
                break
            time.sleep(0.2)
    receipt['members_after'] = owned_group_members(expected)
    receipt['exit_code'] = process.poll()
    if receipt['members_after']:
        write(OPS / 'CLEANUP_FAILED.json', receipt)
        raise RuntimeError('Owned descendants remain after bounded termination')
    return receipt


def tail(path, limit=131072):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def progress(run):
    stream = run / 'solver_calls.jsonl'
    rows = []
    for line in tail(stream).splitlines():
        try:
            row = json.loads(line)
            if row.get('global_step') is not None:
                rows.append(row)
        except ValueError:
            pass
    return {'step': max((r['global_step'] for r in rows), default=0),
            'last_telemetry_age_seconds': time.time() - stream.stat().st_mtime if stream.exists() else None,
            'last_solver_call': rows[-1] if rows else None}


def preflight_gate():
    contract = binding()
    if (OPS / 'PREFLIGHT_FAILED.json').exists():
        raise RuntimeError('Preflight FAILED takes precedence over DONE')
    receipt = read(OPS / 'PREFLIGHT_DONE.json')
    if any(receipt.get(k) != v for k, v in contract.items()):
        raise RuntimeError('Preflight provenance mismatch')
    if receipt.get('counts') != {'smokes': 4, 'smoke_evaluations': 4, 'pilots': 4}:
        raise RuntimeError('Incomplete preflight counts')
    for name, expected in receipt['evidence_sha256'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f'Preflight evidence mismatch: {name}')


def health_gate():
    if (OPS / 'MONITOR_FAILED.json').exists() or (OPS / 'ADMISSION_BLOCKED.json').exists():
        raise RuntimeError('Monitor failure or global admission block')
    live = read(OPS / 'campaign_live.json')
    if time.time() - live['observed_at'] > 120:
        raise RuntimeError('Stale coordinator monitor')
    if live.get('contract_sha256') != os.environ['DEFLOW_CONTRACT_SHA256']:
        raise RuntimeError('Monitor belongs to another contract')


def sync_gate(required=None, final_hashes=None):
    expected = binding()
    while True:
        latch_check()
        if not final_hashes:
            health_gate()
        try:
            receipt = read(OPS / 'sync_receipt.json')
            age = time.time() - receipt['at']
            valid = (all(receipt.get(k) == v for k, v in expected.items())
                     and receipt['result'] == 'ok' and 0 <= age < 900
                     and receipt.get('index_sha256')
                     and (required is None or required in receipt['accepted_attempts'])
                     and (not final_hashes or all(receipt['verified_sha256'].get(k) == v for k, v in final_hashes.items())))
            if valid and required is not None:
                accepted_receipt = read(OPS / 'attempts' / required / 'ACCEPTED.json')
                valid = all(receipt['verified_sha256'].get(name) == sha
                            for name, sha in accepted_receipt['artifacts_sha256'].items())
            if valid:
                contract_check()
                latch_check()
                if not final_hashes:
                    health_gate()
                return
        except (FileNotFoundError, ValueError, KeyError):
            pass
        SHUTDOWN.wait(5)


def execute(run_id, argv, slot, run=None, scientific=False, required=None):
    latch_check()
    check_sources()
    if scientific:
        preflight_gate()
        sync_gate(required)
    lease = (OPS / f'gpu_{slot}.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = OPS / 'attempts' / run_id
    process = None
    owned = None
    started = time.time()
    try:
        latch_check()
        observation = capacity(slot)
        attempt.mkdir(parents=True, exist_ok=False)
        log = OPS / 'logs' / (run_id + '.log')
        log.parent.mkdir(exist_ok=True)
        contract = dict(binding(), run_id=run_id, attempt_id=0, argv=argv, cwd=str(PACKAGE),
            gpu_slot=slot, gpu_uuid=env(slot)['CUDA_VISIBLE_DEVICES'],
            environment={k: env(slot)[k] for k in ('PYTHONPATH', 'DEFLOW_ENGINE', 'CUDA_VISIBLE_DEVICES')},
            command_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest(),
            claimed_at=started, capacity_sample=observation, log=str(log),
            run_path=str(run) if run else None, retry_budget=0, resume=False)
        write(attempt / 'CLAIM.json', contract)
        check_sources()
        latch_check()
        if scientific:
            preflight_gate()
            health_gate()
        with log.open('x') as stream:
            process = subprocess.Popen(argv, cwd=PACKAGE, env=env(slot), stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                owned = identity(process.pid)
            except FileNotFoundError:
                if process.poll() is None:
                    raise
            contract.update(identity=owned, pid=process.pid, started_at=started, state='RUNNING')
            write(attempt / 'START.json', contract)
            with LOCK:
                ACTIVE[run_id] = contract
            while process.poll() is None:
                if stop_requested():
                    write(attempt / 'STOP_RECEIPT.json', terminate_owned(process, owned))
                    raise RuntimeError('Owned child stopped by campaign STOP')
                write(attempt / 'LIVE.json', dict(contract, observed_at=time.time(), telemetry=progress(run) if run else {}))
                SHUTDOWN.wait(2)
            code = process.wait()
            if owned is not None and owned_group_members(owned):
                write(attempt / 'STOP_RECEIPT.json', terminate_owned(process, owned))
                raise RuntimeError('Child leader exited with lingering descendants')
        terminal = dict(contract, exit_code=code, finished_at=time.time(),
                        elapsed_seconds=time.time() - started, state='EXITED')
        write(attempt / 'EXIT.json', terminal)
        if code:
            raise RuntimeError(f'{run_id} exited {code}. See {log}')
        return terminal
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:
                write(attempt / 'STOP_RECEIPT.json', terminate_owned(process, owned))
            except BaseException as cleanup_error:
                write(OPS / 'CLEANUP_FAILED.json', dict(at=time.time(), run_id=run_id,
                    identity=owned, error=repr(cleanup_error)))
                raise
        if attempt.exists():
            write(attempt / 'FAILED.json', dict(at=time.time(), run_id=run_id,
                state='INTERRUPTED' if stop_requested() else 'FAILED', error=repr(error),
                exit_code=process.returncode if process else None))
        raise
    finally:
        with LOCK:
            ACTIVE.pop(run_id, None)
        lease.close()


def train_argv(job, output, steps=None, smoke=False, plan=False):
    argv = [str(a).replace('{REPO}', str(REPO)).replace('{OUTPUT}', str(OUTPUT)) for a in job['argv']]
    argv[0] = PYTHON
    argv[argv.index('--output') + 1] = str(output)
    if steps is not None:
        argv[argv.index('--steps') + 1] = str(steps)
    if smoke:
        argv.append('--smoke')
    if plan:
        argv[argv.index('--execute')] = '--plan'
    return argv + ['--precision-policy', 'strict_fp32']


def make_plan(job, run, label, steps=None, smoke=False):
    latch_check()
    contract_check()
    result = subprocess.run(train_argv(job, run, steps, smoke, True), cwd=PACKAGE,
        env=env(job['gpu_slot']), capture_output=True, text=True, check=True, timeout=120)
    write(OPS / 'plans' / (label + '.json'), json.loads(result.stdout))


def validate_precision(run, training=False):
    receipt = read(run / 'runtime_precision.json')
    if receipt.get('policy') != 'strict_fp32':
        raise RuntimeError(f'Wrong actual precision policy: {run}')
    required = {'actor_initialization', 'actor_forward', 'solver_entry', 'solver_result'}
    if training:
        required |= {'density_entry', 'density_result', 'backward'}
    if not required <= receipt.get('events', {}).keys():
        raise RuntimeError(f'Missing actual precision boundaries: {run}')
    for name in required:
        event = receipt['events'][name]
        actual = event['actual']
        expected = {'matmul_precision': 'highest', 'matmul_allow_tf32': False,
                    'cudnn_allow_tf32': False, 'autocast_cuda': False,
                    'autocast_cpu': False, 'default_dtype': 'torch.float32',
                    'torch': '2.5.1+cu124', 'cuda': '12.4'}
        if actual.get('backend_environment') or any(actual.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f'Actual precision mismatch at {name}: {run}')
        if name != 'actor_initialization':
            if not event['devices'] or any(not str(d).startswith('cuda') for d in event['devices']):
                raise RuntimeError(f'Actual CUDA not confirmed at {name}: {run}')
            if any(d != 'torch.float32' for d in event['dtypes']):
                raise RuntimeError(f'Non-FP32 tensor at {name}: {run}')
    return receipt


def validate_run(run, horizon, batch):
    import torch
    done = read(run / 'DONE.json')
    if done['exit_code'] != 0 or (run / 'FAILED.json').exists():
        raise RuntimeError(f'Conflicting training terminal evidence: {run}')
    candidates = [c for c in done['checkpoints'] if horizon <= c['global_step'] <= horizon + 7]
    if not candidates:
        raise RuntimeError(f'Missing required horizon: {run}')
    checkpoint = Path(candidates[-1]['path'])
    if not checkpoint.resolve().is_relative_to(run.resolve()):
        raise RuntimeError('Checkpoint escapes run root')
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if data['global_step'] != candidates[-1]['global_step'] or data['args']['batch_size'] != batch:
        raise RuntimeError('Checkpoint step or batch mismatch')
    for key in ('actor', 'q1', 'q2', 'log_alpha'):
        values = data[key].values() if isinstance(data[key], dict) else [data[key]]
        if not all(torch.isfinite(tensor).all() for tensor in values):
            raise RuntimeError(f'Nonfinite checkpoint: {key}')
    for name in ('manifest.json', 'actor_kwargs.json', 'solver_calls.jsonl', 'telemetry_summary.json'):
        if not (run / name).stat().st_size:
            raise RuntimeError(f'Missing artifact: {name}')
    if read(run / 'telemetry_summary.json')['roles'].get('training/actor_grad', {}).get('calls', 0) < 1:
        raise RuntimeError('No actor updates')
    if not next(run.glob('capsule_*.pt'), None):
        raise RuntimeError('No capsules')
    validate_precision(run, training=True)
    return checkpoint


def accepted(run_id, paths):
    hashes = {str(Path(path).relative_to(ROOT)): digest(path) for path in paths}
    write(OPS / 'attempts' / run_id / 'ACCEPTED.json', dict(binding(), run_id=run_id,
        at=time.time(), artifacts_sha256=hashes))


def evaluate(run_id, run, checkpoint, output, episodes, slot, scientific=False):
    result = execute(run_id, [PYTHON, '-m', 'deflow_rebuttal.evaluate', '--repo', str(REPO),
        '--run', str(run), '--checkpoint', str(checkpoint), '--output', str(output),
        '--episodes', str(episodes), '--eval-seed', '26152', '--device', 'cuda'], slot, output, scientific)
    evaluation = read(output / 'evaluation.json')
    if len(evaluation['episodes']) != episodes or evaluation['eval_seed'] != 26152:
        raise RuntimeError('Evaluation protocol mismatch')
    if evaluation['checkpoint_sha256'] != digest(checkpoint):
        raise RuntimeError('Evaluation checkpoint hash mismatch')
    import math
    if not all(math.isfinite(evaluation['summary'][k]) for k in ('u4', 'total_success', 'mean_return')):
        raise RuntimeError('Missing/nonfinite evaluation metrics')
    validate_precision(output)
    accepted(run_id, [checkpoint, run / 'DONE.json', run / 'manifest.json',
        run / 'runtime_precision.json', run / 'actor_kwargs.json', run / 'telemetry_summary.json',
        output / 'evaluation.json', output / 'runtime_precision.json'])
    return result
