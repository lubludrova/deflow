"""Launch one E03 case at a time and verify its full local copy before advancing."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time


BASE = Path(__file__).resolve().parents[1]
RUN = BASE / 'runs/tol1e-6-cap100-20261112'
MANIFEST = json.loads((RUN / 'manifest.json').read_text())
PROFILE = json.loads((BASE.parents[1] / 'common/operations/active_campaign.json').read_text())
REMOTE = MANIFEST['remote_root']
PYTHON = MANIFEST['source_root'] + '/venv/bin/python'
SSH = [PROFILE['ssh_wrapper'], PROFILE['server']['user'] + '@' + PROFILE['server']['host']]
ANCHOR = hashlib.sha256((RUN / 'manifest.json').read_bytes()).hexdigest()


def remote(program):
    return json.loads(subprocess.check_output(SSH + ['python3 -'], input=program,
                                             text=True, timeout=55))


def main():
    for job in MANIFEST['jobs']:
        name = job['run_id']
        print('START', name, flush=True)
        launch = remote(f'''
from pathlib import Path
import json,subprocess
r=Path({REMOTE!r})
with (r/({name!r}+'.supervisor.log')).open('x') as log:
 p=subprocess.Popen({[PYTHON, REMOTE+'/code/run_case.py', REMOTE+'/manifest.json', ANCHOR, name]!r},
     cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))
''')
        (RUN / (name + '.launch.json')).write_text(json.dumps(launch, indent=2) + '\n')
        while True:
            time.sleep(5)
            status = remote(f'''
from pathlib import Path
import json,sys
sys.path.insert(0,{MANIFEST['source_root']!r}+'/ops')
import common as c
p=Path({REMOTE!r})/'attempts'/{name!r}
d={{'state':'STARTING'}}
if (p/'START.json').exists():
 s=c.read(p/'START.json')
 d.update(state='RUNNING',child_identity_matches=c.same_process(s['identity']),supervisor_alive=c.same_process(s['supervisor']))
if (p/'FAILED.json').exists():d.update(state='FAILED',failure=c.read(p/'FAILED.json'))
elif (p/'ACCEPTED.json').exists():d.update(state='ACCEPTED',accepted=c.read(p/'ACCEPTED.json'))
elif (p/'EXIT.json').exists():d['state']='EXITED'
print(json.dumps(d))
''')
            if status['state'] == 'FAILED':
                raise RuntimeError(status)
            if status['state'] == 'ACCEPTED' and not status.get('supervisor_alive', True):
                break
        members = [f'audits/{name}', f'attempts/{name}', name + '.supervisor.log']
        command = shlex.join(['tar', '-C', REMOTE, '-czf', '-', *members])
        archive = subprocess.check_output(SSH + [command], timeout=55)
        mirror = RUN / 'mirror'
        mirror.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            for member in tar.getmembers():
                target = mirror / member.name
                if member.issym() or member.islnk() or not target.resolve().is_relative_to(mirror.resolve()):
                    raise RuntimeError('Unexpected archive member')
                if member.isfile() and target.exists():
                    raise RuntimeError('Refusing to overwrite a prior local result')
            tar.extractall(mirror)
        hashes = remote(f'''
from pathlib import Path
import hashlib,json
r=Path({REMOTE!r})
files=[]
for relative in {members!r}:
 p=r/relative
 files.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()])
print(json.dumps({{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}))
''')
        for relative, expected in hashes.items():
            if hashlib.sha256((mirror / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Synchronization hash mismatch: ' + relative)
        for relative, expected in status['accepted']['artifacts'].items():
            if hashes.get(relative) != expected:
                raise RuntimeError('Accepted result hash changed: ' + relative)
        receipt = dict(run_id=name, verified_at=time.time(), sha256=hashes,
                       reference_valid=status['accepted']['reference_valid'],
                       reference_invalid=status['accepted']['reference_invalid'])
        (RUN / (name + '.sync.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        print('VERIFIED', name, 'valid', receipt['reference_valid'],
              'invalid', receipt['reference_invalid'], flush=True)
    print('ALL_NINE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
