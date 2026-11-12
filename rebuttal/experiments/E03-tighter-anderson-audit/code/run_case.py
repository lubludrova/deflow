"""Run one authorized E03 audit using the unchanged E01 scientific implementation."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    manifest_path, anchor, run_id = sys.argv[1:4]
    root = Path(manifest_path).parent
    manifest = json.loads(Path(manifest_path).read_text())
    sys.path.insert(0, manifest['source_root'] + '/ops')
    import common as c

    if c.digest(manifest_path) != anchor or c.digest(__file__) != manifest['runner_sha256']:
        raise RuntimeError('E03 manifest or runner hash mismatch')
    os.environ['DEFLOW_CONTRACT_SHA256'] = manifest['source_contract_sha256']
    c.contract_check()
    c.latch_check(wait_pause=False)
    job = next(row for row in manifest['jobs'] if row['run_id'] == run_id)
    if c.digest(job['capsule']) != job['capsule_sha256']:
        raise RuntimeError('Frozen input capsule hash mismatch')
    if '--check' in sys.argv[4:]:
        print(json.dumps({'run_id': run_id, 'source_and_input_verified': True}))
        return
    attempt = root / 'attempts' / run_id
    attempt.mkdir(parents=True, exist_ok=False)
    record = dict(job, manifest_sha256=anchor, attempt_id=0, retry_budget=0,
                  supervisor=c.identity(os.getpid()), started_at=time.time())
    c.write(attempt / 'CLAIM.json', record)
    process = owned = None
    try:
        with contextlib.ExitStack() as stack:
            # Share the active coordinator's admission and physical-GPU lane locks.
            for lane in (0, 1):
                lock = stack.enter_context((Path(manifest['admission_root']) /
                                            f'gpu_1_lane_{lane}.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (Path(manifest['admission_root']) / 'admission.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                record['capacity_before'] = c.capacity(1)
                env = c.env(1)
                with (attempt / 'audit.log').open('x') as log:
                    process = subprocess.Popen(job['argv'], cwd=manifest['cwd'], env=env,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                for _ in range(100):
                    owned = c.identity(process.pid)
                    if owned['cmdline'].strip() == ' '.join(job['argv']):
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError('Child command identity did not settle')
                record['identity'] = owned
                record['environment'] = {k: env[k] for k in manifest['environment_keys']}
                c.write(attempt / 'START.json', record)
            code = process.wait()
            c.write(attempt / 'EXIT.json', dict(record, exit_code=code,
                    finished_at=time.time(), elapsed_seconds=time.time()-record['started_at']))
            if code:
                raise RuntimeError(f'Audit exited {code}')
            if c.owned_group_members(owned):
                raise RuntimeError('Audit descendants remain after exit')
            output = Path(job['output'])
            summary = c.read(output / 'summary.json')
            if (summary['batch'] != 1024 or summary['precision_policy'] != 'strict_fp32'
                    or summary['effective_anderson'] != {'tol': 1e-6, 'cap': 100}
                    or summary['loss_kind'] != 'configured_SAC_actor_loss'
                    or summary['hashes']['inputs'] != job['capsule_sha256']):
                raise RuntimeError('Audit protocol mismatch')
            files = [output / 'summary.json', output / 'raw.pt']
            c.write(attempt / 'ACCEPTED.json', dict(run_id=run_id, accepted_at=time.time(),
                    artifacts={str(p.relative_to(root)): c.digest(p) for p in files},
                    reference_valid=summary['reference_valid'],
                    reference_invalid=summary['reference_invalid']))
    except BaseException:
        if process is not None and process.poll() is None and owned is not None:
            c.write(attempt / 'STOP_RECEIPT.json', c.terminate_owned(process, owned))
        c.write(attempt / 'FAILED.json', dict(run_id=run_id, at=time.time(),
                traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
