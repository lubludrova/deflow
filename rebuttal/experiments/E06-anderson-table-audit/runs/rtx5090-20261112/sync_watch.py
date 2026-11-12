"""Copy each accepted audit and ACK only exact local SHA-256 matches."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
REMOTE = MANIFEST['remote_root']
SSH = ['ssh', '-o', 'ControlPath=/tmp/deflow-sa-20261112.sock', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
       '-p', '13534', 'featurize@workspace.featurize.cn']


def remote(program):
    return subprocess.check_output(SSH + ['python3 -'], input=program, text=True, timeout=55)


def main():
    for job in MANIFEST['jobs']:
        rid = job['run_id']
        while True:
            try:
                result = json.loads(remote(f'''from pathlib import Path
import json
r=Path({REMOTE!r});p=r/'attempts'/{rid!r}/'ACCEPTED.json'
print(json.dumps({'failure': json.loads((r/'FAILED.json').read_text())}) if (r/'FAILED.json').exists() else (p.read_text() if p.exists() else 'null'))
'''))
                if result is not None and 'failure' in result:
                    raise RuntimeError(result['failure'])
                if result is None:
                    time.sleep(10)
                    continue
                mirror = ROOT / 'mirror'
                mirror.mkdir(exist_ok=True)
                members = ['audits/' + rid, 'attempts/' + rid]
                command = shlex.join(['tar', '-C', REMOTE, '-cf', '-', *members])
                producer = subprocess.Popen(SSH + [command], stdout=subprocess.PIPE)
                consumer = subprocess.run(['tar', '-xf', '-', '-C', str(mirror)], stdin=producer.stdout)
                producer.stdout.close()
                assert producer.wait() == consumer.returncode == 0
                for rel, expected in result['sha256'].items():
                    assert hashlib.sha256((mirror / rel).read_bytes()).hexdigest() == expected, rel
                ack = {'at': time.time(), 'sha256': result['sha256']}
                (ROOT / (rid + '.sync.json')).write_text(json.dumps(ack, indent=2))
                remote(f'''from pathlib import Path
import json
p=Path({REMOTE!r})/'attempts'/{rid!r}/'SYNC_ACK.json'
value={ack!r}
if p.exists():assert json.loads(p.read_text())['sha256']==value['sha256']
else:
 with p.open('x') as f:json.dump(value,f)
''')
                print('VERIFIED', rid, flush=True)
                break
            except (subprocess.SubprocessError, OSError) as error:
                print('TRANSPORT_WAIT', type(error).__name__, flush=True)
                time.sleep(20)
    print('ALL_18_SYNCED', flush=True)


if __name__ == '__main__':
    main()
