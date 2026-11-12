"""120-second compact pull, anchored provenance and immutable artifact verification."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from common import write, digest

LOCAL = Path(__file__).resolve().parent.parent
OPS = LOCAL / 'ops'


def terminal_ready(mirror, verified):
    final_ack = mirror / 'ops/FINAL_SYNC_ACK.json'
    preflight_failed = 'ops/PREFLIGHT_FAILED.json' in verified
    preflight_quiescent = 'ops/PREFLIGHT_TERMINAL_FAILED.json' in verified
    coordinator_started = (mirror / 'ops/COORDINATOR_STARTED.json').exists()
    coordinator_failed = 'ops/COORDINATOR_FAILED.json' in verified
    if final_ack.exists() and 'ops/FINAL_SYNC_ACK.json' in verified:
        acknowledged = json.loads(final_ack.read_text())
        if not all(verified.get(k) == v for k, v in acknowledged['final_sha256'].items()):
            raise RuntimeError('Final acknowledgment does not cover local final artifacts')
        terminal = True
    else:
        terminal = ((preflight_failed and preflight_quiescent
                     and (not coordinator_started or coordinator_failed))
                    or (coordinator_failed and 'ops/PREFLIGHT_DONE.json' in verified))
    if (mirror / 'ops/CLEANUP_FAILED.json').exists():
        terminal = False
    return terminal


def main():
    lease = (OPS / 'sync_local.lock').open('a')
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    anchor = os.environ['DEFLOW_CONTRACT_SHA256']
    if digest(OPS / 'execution_contract.json') != anchor:
        raise RuntimeError('Local contract anchor mismatch')
    contract = json.loads((OPS / 'execution_contract.json').read_text())
    for name in ('ops/sync_local.py', 'ops/common.py'):
        if digest(LOCAL / name) != contract['frozen_files'][name]:
            raise RuntimeError('Local sync code differs from frozen contract')
    config = contract['sync']
    remote = contract['remote_root']
    ssh = config['ssh_wrapper']
    host = contract['server']['user'] + '@' + contract['server']['host']
    mirror = Path(config['local_root'])
    import shlex
    command = ' '.join(['env', 'DEFLOW_CONTRACT_SHA256=' + shlex.quote(anchor),
                        shlex.quote(contract['python']), shlex.quote(remote + '/ops/sync_index.py')])
    encoded = subprocess.check_output([ssh, host, command], timeout=600)
    index = json.loads(encoded)
    expected = {'campaign_id': contract['campaign_id'], 'contract_sha256': anchor,
                'precision_policy': 'strict_fp32'}
    if any(index.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Remote index binding mismatch')
    for name in index['files']:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Unsafe index path')
    if shutil.disk_usage(LOCAL).free < config['min_local_free_gib'] * 1024**3:
        raise RuntimeError('Local disk reserve reached. Queue will wait.')
    if index['selected_bytes'] > config['max_mirror_gib'] * 1024**3:
        raise RuntimeError('Compact mirror budget exceeded. Originals retained remotely.')
    files_from = OPS / 'sync-files.txt'
    files_from.write_text('\n'.join(index['files']) + '\n')
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run(['/usr/bin/rsync', '-a', '--timeout=180', '--files-from=' + str(files_from),
        '-e', ssh, host + ':' + remote + '/', str(mirror) + '/'], check=True, timeout=900)
    verified = {}
    for name, expected_hash in index['sha256'].items():
        if digest(mirror / name) != expected_hash:
            raise RuntimeError(f'Local immutable checksum mismatch: {name}')
        verified[name] = expected_hash
    accepted_attempts = []
    for run_id, required in index['accepted_attempts'].items():
        if not all(verified.get(name) == sha for name, sha in required.items()):
            raise RuntimeError(f'Incomplete accepted attempt: {run_id}')
        accepted_attempts.append(run_id)
    receipt = dict(expected, at=time.time(), result='ok', index_sha256=hashlib.sha256(encoded).hexdigest(),
        files=len(index['files']), selected_bytes=index['selected_bytes'], verified_sha256=verified,
        accepted_attempts=accepted_attempts, local_free_bytes=shutil.disk_usage(LOCAL).free)
    # Persist the exact indexed bytes locally for independent receipt reconstruction.
    (OPS / 'sync_index.json').write_bytes(encoded)
    script = ('import os,sys,tempfile,pathlib; p=pathlib.Path(' + repr(remote + '/ops/sync_receipt.json') + '); '
              'fd,q=tempfile.mkstemp(dir=p.parent,suffix=".tmp"); '
              'f=os.fdopen(fd,"w"); f.write(sys.stdin.read()); f.close(); os.replace(q,p)')
    subprocess.run([ssh, host, shlex.join([contract['python'], '-c', script])],
        input=json.dumps(receipt) + '\n', text=True, check=True, timeout=60)
    write(OPS / 'sync_status.json', receipt)
    terminal = terminal_ready(mirror, verified)
    if terminal:
        write(OPS / 'SYNC_TERMINAL.json', dict(receipt, stopped_at=time.time()))
        label = config.get('launchd_label')
        if label:
            subprocess.run(['/bin/launchctl', 'bootout', f'gui/{os.getuid()}/{label}'], check=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(OPS / 'sync_status.json', dict(at=time.time(), result='error', traceback=traceback.format_exc()))
        raise
