"""Frozen, fresh-only three-seed supervisor. No retries, resume, or GPU packing."""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def contract():
    data = json.loads((ROOT/'ops/manifest.json').read_text())
    assert str(ROOT) == data['remote_root']
    for name, expected in data['frozen_files'].items():
        assert digest(ROOT/name) == expected, f'Source changed: {name}'
    return data


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    command = Path(f'/proc/{pid}/cmdline').read_bytes()
    return dict(pid=pid, start_ticks=stat[19], command_sha256=hashlib.sha256(command).hexdigest(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def gpu_snapshot():
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free',
                                  '--format=csv,noheader,nounits'],text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                        '--format=csv,noheader,nounits'],text=True)
    return dict(time=now(),gpus=gpu.strip().splitlines(),processes=processes.strip().splitlines())


def check_peers(snapshot, gpu=None):
    data=contract()
    peer=Path(data['peer']['remote_root'])
    assert digest(peer/'ops/manifest.json')==data['peer']['manifest_sha256']
    pm=json.loads((peer/'ops/manifest.json').read_text())
    allowed={}
    for job in pm['jobs']:
        if gpu is not None and job['gpu']!=gpu: continue
        attempt=peer/'attempts'/job['run_id']/'0000'
        for phase in ('training','evaluation'):
            path=attempt/f'{phase}_launch.json'
            if not path.exists(): continue
            launch=json.loads(path.read_text())
            ident=launch['identity']
            if Path(f"/proc/{ident['pid']}").exists():
                assert identity(ident['pid'])==ident, 'Peer identity changed'
                allowed[str(ident['pid'])]=job['gpu_uuid']
    for row in snapshot['processes']:
        pid,uuid,memory=[x.strip() for x in row.split(',')]
        if gpu is not None and uuid!=data['gpu_by_slot'][str(gpu)]: continue
        assert allowed.get(pid)==uuid, 'Unknown or excess GPU process'
    return allowed


def wait_peer_evaluation(job, attempt):
    data=contract()
    peer=Path(data['peer']['remote_root'])
    pm=json.loads((peer/'ops/manifest.json').read_text())
    pj=next(x for x in pm['jobs'] if x['gpu']==job['gpu'])
    pa=peer/'attempts'/pj['run_id']/'0000'
    while True:
        snapshot=gpu_snapshot()
        check_peers(snapshot,job['gpu'])
        terminal=any((pa/name).exists() for name in ('EVALUATION_ACCEPTED.json','FAILED.json'))
        free=not any(job['gpu_uuid'] in row for row in snapshot['processes'])
        if terminal and free: return
        write(attempt/'waiting_peer.json',dict(time=now(),peer=pj['run_id'],peer_terminal=terminal,gpu_free=free))
        time.sleep(30)


def run_child(job, phase, argv, attempt):
    data = contract()
    environment = os.environ.copy()
    environment.update(data['environment'])
    environment['CUDA_VISIBLE_DEVICES'] = job['gpu_uuid']
    logfile = attempt/f'{phase}.log'
    started = time.time()
    with logfile.open('x') as output:
        child = subprocess.Popen(argv,cwd=ROOT,env=environment,stdout=output,stderr=subprocess.STDOUT)
        expected = hashlib.sha256(b''.join(os.fsencode(x)+b'\0' for x in argv)).hexdigest()
        for _ in range(100):
            ident = identity(child.pid)
            if ident['command_sha256'] == expected or child.poll() is not None:
                break
            time.sleep(0.01)
        if child.poll() is None and ident['command_sha256'] != expected:
            write(attempt/f'{phase}_identity_warning.json', dict(time=now(), identity=ident, expected=expected))
        write(attempt/f'{phase}_launch.json',dict(time=now(),identity=ident,argv=argv,
              gpu=job['gpu'],gpu_uuid=job['gpu_uuid'],manifest_sha256=digest(ROOT/'ops/manifest.json')))
        while child.poll() is None:
            with logfile.open('rb') as source:
                source.seek(max(0,logfile.stat().st_size-131072))
                tail=source.read().decode(errors='replace')
            steps=re.findall(r'\bstep\s*[=:]\s*(\d+)',tail)
            disk=shutil.disk_usage(ROOT)
            write(attempt/'live.json',dict(time=now(),phase=phase,identity=ident,
                  step=int(steps[-1]) if steps else None,elapsed_seconds=time.time()-started,
                  log_modified=logfile.stat().st_mtime,disk_used_fraction=disk.used/disk.total,
                  gpu=gpu_snapshot()))
            time.sleep(30)
        receipt=dict(time=now(),exit_code=child.returncode,identity=ident,
                     elapsed_seconds=time.time()-started,argv=argv)
        write(attempt/f'{phase}_exit.json',receipt)
    if child.returncode:
        raise RuntimeError(f'{phase} exited {child.returncode}, no retry')
    return receipt


def wait_sync(attempt, required_checkpoint=None):
    write(attempt/'waiting_sync.json',dict(time=now(),checkpoint=required_checkpoint))
    while True:
        path=ROOT/'ops/SYNC_ACK.json'
        if path.exists():
            ack=json.loads(path.read_text())
            if (ack.get('manifest_sha256')==digest(ROOT/'ops/manifest.json')
                and time.time()-ack['timestamp'] < 600) and (
                required_checkpoint is None or ack.get('critical_sha256',{}).get(required_checkpoint)==digest(ROOT/required_checkpoint)):
                return
        time.sleep(30)


def worker(run_id):
    data=contract()
    job=next(j for j in data['jobs'] if j['run_id']==run_id)
    attempt=ROOT/'attempts'/run_id/'0000'
    attempt.mkdir(parents=True,exist_ok=False)
    write(attempt/'claim.json',dict(time=now(),run_id=run_id,attempt_id='0000',
          supervisor=identity(os.getpid()),manifest_sha256=digest(ROOT/'ops/manifest.json')))
    try:
        wait_sync(attempt)
        snapshot=gpu_snapshot()
        check_peers(snapshot,job['gpu'])
        assert shutil.disk_usage(ROOT).free > 30*1024**3, 'Insufficient free disk'
        run_child(job,'training',job['train_argv'],attempt)
        result=ROOT/'results'/run_id
        done=json.loads((result/'DONE.json').read_text())
        assert not (result/'FAILED.json').exists()
        candidates=[x for x in done['checkpoints'] if 200000<=x['global_step']<=200007]
        assert candidates, 'Missing expected-horizon checkpoint'
        checkpoint=next(Path(x['path']) for x in candidates if Path(x['path']).name.startswith('ckpt_step_'))
        import torch
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert 200000<=payload['global_step']<=200007
        assert payload['args']['seed']==job['seed']
        for name in ('actor','q1','q2'):
            assert all(bool(torch.isfinite(t).all()) for t in payload[name].values())
        assert bool(torch.isfinite(payload['log_alpha']).all())
        rel=str(checkpoint.relative_to(ROOT))
        write(attempt/'TRAINING_ACCEPTED.json',dict(time=now(),step=payload['global_step'],
              checkpoint=rel,sha256=digest(checkpoint)))
        wait_sync(attempt,rel)
        wait_peer_evaluation(job,attempt)
        assert not any(job['gpu_uuid'] in row for row in gpu_snapshot()['processes']), 'GPU occupied before evaluation'
        assert shutil.disk_usage(ROOT).free > 30*1024**3, 'Insufficient free disk'
        argv=[value.replace('{CHECKPOINT}',str(checkpoint)) for value in job['eval_argv']]
        run_child(job,'evaluation',argv,attempt)
        evaluation=ROOT/'evaluations'/run_id/'evaluation.json'
        summary=json.loads(evaluation.read_text())
        assert len(summary['episodes'])==1000 and summary['eval_seed']==26152
        assert summary['checkpoint_sha256']==digest(checkpoint)
        write(attempt/'EVALUATION_ACCEPTED.json',dict(time=now(),sha256=digest(evaluation),summary=summary['summary']))
    except BaseException:
        write(attempt/'FAILED.json',dict(time=now(),traceback=traceback.format_exc()))
        raise


def start():
    lease = open('/home/featurize/.deflow-gpu-admission.lock', 'a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data=contract()
    assert not (ROOT/'ops/STARTED.json').exists(), 'Campaign already submitted'
    for gate, expected in data['release_gates'].items():
        assert digest(ROOT/gate)==expected, f'Missing or changed release gate {gate}'
    snapshots=[]
    for i in range(7):
        snapshot=gpu_snapshot()
        check_peers(snapshot)
        assert len(snapshot['gpus'])==3
        for row in snapshot['gpus']:
            index,uuid,util,used,free=[x.strip() for x in row.split(',')]
            assert uuid==data['gpu_by_slot'][index]
            assert int(free)>12000
        snapshots.append(snapshot)
        if i<6: time.sleep(10)
    write(ROOT/'ops/admission.json',dict(time=now(),samples=snapshots))
    wait_sync(ROOT/'ops')
    with (ROOT/'ops/STARTED.json').open('x') as f:
        json.dump(dict(time=now(),manifest_sha256=digest(ROOT/'ops/manifest.json')),f)
    for job in data['jobs']:
        contract()
        subprocess.run(['tmux','new-session','-d','-s',job['session'],
                        sys.executable,str(Path(__file__).resolve()),'--worker',job['run_id']],check=True)
    print(json.dumps(dict(submitted=[j['run_id'] for j in data['jobs']]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',action='store_true')
    parser.add_argument('--worker')
    args=parser.parse_args()
    if args.start: start()
    elif args.worker: worker(args.worker)
