"""Run the approved fixed queue once, with real exits and verified sync gates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert sha(ROOT / 'manifest.json') == sys.argv[1]
    assert sha(Path(__file__)) == manifest['supervisor_sha256']
    for relative, expected in manifest['frozen_files'].items():
        assert sha(ROOT / relative) == expected, relative
    for job in manifest['jobs']:
        assert sha(ROOT / job['input']) == job['input_sha256']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(manifest['gpu']),
               PYTHONPATH=str(ROOT / 'code'), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    write(ROOT / 'CLAIM.json', {'pid': os.getpid(), 'at': time.time(), 'manifest_sha256': sys.argv[1]})
    try:
        for job in manifest['jobs']:
            attempt = ROOT / 'attempts' / job['run_id']
            attempt.mkdir(parents=True, exist_ok=False)
            samples = []
            for _ in range(3):
                s = subprocess.check_output(['nvidia-smi', '-i', str(manifest['gpu']),
                    '--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
                fields = [x.strip() for x in s.split(',')]
                assert int(fields[1]) == 0 and int(fields[2]) < 100, s
                samples.append(s)
                time.sleep(2)
            write(attempt / 'CLAIM.json', dict(job, gpu_samples=samples, at=time.time()))
            start = time.time()
            with (attempt / 'audit.log').open('x') as log:
                process = subprocess.Popen(job['argv'], cwd=ROOT / 'code', env=env,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                write(attempt / 'START.json', {'pid': process.pid, 'argv': job['argv'], 'at': start})
                code = process.wait()
            write(attempt / 'EXIT.json', {'exit_code': code, 'at': time.time(), 'elapsed_seconds': time.time()-start})
            assert code == 0, (job['run_id'], code)
            output = ROOT / 'audits' / job['run_id']
            summary = json.loads((output / 'summary.json').read_text())
            assert summary['batch'] == 1024 and summary['precision_policy'] == 'strict_fp32'
            assert summary['anderson_sweep_caps'] == job['caps']
            assert summary['effective_anderson']['tol'] == job['tol']
            assert summary['hashes']['inputs'] == job['input_sha256']
            assert summary['loss_kind'] == 'configured_SAC_actor_loss'
            files = [p for folder in (attempt, output) for p in folder.rglob('*') if p.is_file()]
            hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
            write(attempt / 'ACCEPTED.json', {'at': time.time(), 'sha256': hashes})
            print('ACCEPTED', job['run_id'], flush=True)
            while not (attempt / 'SYNC_ACK.json').exists():
                time.sleep(3)
            ack = json.loads((attempt / 'SYNC_ACK.json').read_text())
            assert ack['sha256'] == hashes
        write(ROOT / 'COMPLETED.json', {'at': time.time(), 'jobs': len(manifest['jobs']), 'exit_code': 0})
    except BaseException:
        write(ROOT / 'FAILED.json', {'at': time.time(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
