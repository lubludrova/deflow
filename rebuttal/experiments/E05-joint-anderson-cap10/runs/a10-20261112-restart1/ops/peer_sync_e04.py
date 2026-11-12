"""Compact nondeleting mirror and verified checkpoint acknowledgment."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path('/Users/ruslanageev/dev/DEFlow/rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/ops')
ROOT=HERE.parent
PROFILE=json.loads((HERE/'profile.json').read_text())
SSH=['ssh','-S',PROFILE['control_socket'],'-o','BatchMode=yes','-o','ConnectTimeout=20',
     '-p',str(PROFILE['server']['port']),PROFILE['server']['user']+'@'+PROFILE['server']['host']]


def cycle():
    mirror=ROOT/'mirror'
    mirror.mkdir(exist_ok=True)
    source=PROFILE['remote_root']
    command=' '.join(SSH[:-1])
    subprocess.run(['rsync','-az','--timeout=90','--exclude=venv/','--exclude=code/','--exclude=*.tar*','--exclude=full-backups/',
        '--exclude=validation/inputs/','--exclude=capsule_*.pt','--exclude=solver_calls.jsonl',
        '--exclude=__pycache__/','--exclude=*.tmp','--exclude=ckpt.pt','-e',command,
        SSH[-1]+':'+source+'/',str(mirror)+'/'],check=True)
    critical={}
    for path in (p for p in mirror.rglob('*') if p.is_file() and (p.suffix=='.pt' or p.name in ('evaluation.json','TRAINING_ACCEPTED.json','EVALUATION_ACCEPTED.json','training_exit.json','evaluation_exit.json'))):
        critical[str(path.relative_to(mirror))]=hashlib.sha256(path.read_bytes()).hexdigest()
    program='''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
expected=json.load(sys.stdin)
for relative,digest in expected.items():
 p=root/relative
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest, relative
print(json.dumps(dict(verified=len(expected))))
'''
    import shlex
    response=subprocess.run(SSH+['python3 -c '+shlex.quote(program)+' '+shlex.quote(source)],
                            input=json.dumps(critical),text=True,capture_output=True,check=True)
    manifest=mirror/'ops/manifest.json'
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    peer_root='/home/featurize/deflow-joint-anderson-a10-20261112-restart1'
    guard_program="""import json,sys
from pathlib import Path
root=Path(sys.argv[1])
held=[]
for seed in (1,2,3):
 a=root/'attempts'/f'joint_anderson_a10_s{seed}'/'0000'
 if (root/'ops/STARTED.json').exists() and not (a/'training_exit.json').exists():
  held.append(f'joint_anderson_a100_s{seed}')
print(json.dumps(held))
"""
    held=json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(guard_program)+' '+shlex.quote(peer_root)],text=True))
    verified_critical=critical.copy()
    for relative in list(critical):
        if any('/'+rid+'/' in relative for rid in held) and Path(relative).name.startswith('ckpt_step_'):
            step=int(Path(relative).stem.split('_')[-1])
            if step>=200000: critical.pop(relative)
    (HERE/'SHARED_ACK_GUARD.json').write_text(json.dumps(dict(time=time.time(),held_run_ids=held,verified_artifacts=len(verified_critical)),indent=2)+'\n')
    ack=dict(timestamp=time.time(),campaign_id=PROFILE['campaign_id'],manifest_sha256=manifest_sha256,
             critical_sha256=critical,verification=json.loads(response.stdout),
             scope='compact mirror, all checkpoints/events/receipts, raw solver_calls and capsules retained remotely')
    temporary=HERE/'SYNC_ACK.tmp'
    temporary.write_text(json.dumps(ack,indent=2)+'\n')
    temporary.replace(HERE/'SYNC_ACK.json')
    subprocess.run(['scp','-q','-o','ControlPath='+PROFILE['control_socket'],'-o','BatchMode=yes',
        '-P',str(PROFILE['server']['port']),str(HERE/'SYNC_ACK.json'),
        SSH[-1]+':'+source+'/ops/SYNC_ACK.tmp'],check=True)
    subprocess.run(SSH+['mv '+shlex.quote(source+'/ops/SYNC_ACK.tmp')+' '+shlex.quote(source+'/ops/SYNC_ACK.json')],check=True)
    accepted=list((mirror/'attempts').glob('*/0000/EVALUATION_ACCEPTED.json'))
    if len(accepted)==3:
        (HERE/'FINAL_SYNC_ACK.json').write_text(json.dumps(ack,indent=2)+'\n')
    print(json.dumps(dict(time=time.time(),verified_checkpoints=len(critical),evaluations=len(accepted))),flush=True)


if __name__=='__main__':
    cycle()
