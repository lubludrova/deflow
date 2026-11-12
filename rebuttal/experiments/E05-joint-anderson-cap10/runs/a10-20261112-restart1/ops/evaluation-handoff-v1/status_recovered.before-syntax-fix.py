"""Read-only overlay for first evaluations after administrative handoff."""
import json,subprocess,sys,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=json.loads((HERE/'profile.json').read_text())
s=json.loads(subprocess.check_output([sys.executable,str(HERE/'status.py')],text=True))
program='''import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1]);out={}
for seed in (1,2,3):
 rid=f'joint_anderson_a10_s{seed}';a=root/'ops/evaluation-handoff-v1/attempts'/rid/'0000'
 if not a.exists():continue
 original=root/'attempts'/rid/'0000'
 failure=json.loads((original/'FAILED.json').read_text())
 assert failure['traceback'].rstrip().endswith('AssertionError: Peer identity changed')
 assert not (original/'evaluation_launch.json').exists()
 row={'administrative_failure_retained':str(original/'FAILED.json'),'recovery_attempt':str(a),'state':'CLAIMED'}
 for filename,state in [('evaluation_launch.json','EVALUATION'),('EVALUATION_ACCEPTED.json','EVALUATION_ACCEPTED'),('FAILED.json','FAILED')]:
  if (a/filename).exists():row.update(state=state,receipt=json.loads((a/filename).read_text()))
 out[rid]=row
print(json.dumps(out))
'''
server=p['server']
ssh=['ssh','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',str(server['port']),server['user']+'@'+server['host']
recovered=json.loads(subprocess.check_output(ssh+['python3 -c '+shlex.quote(program)+' '+shlex.quote(p['remote_root'])],text=True))
for j in s['jobs']:
    if 'TRAINING_ACCEPTED.json' in j:j['step']=j['TRAINING_ACCEPTED.json']['step']
    if j['run_id'] in recovered:
        j['evaluation_handoff']=recovered[j['run_id']]
        j['state']=recovered[j['run_id']]['state']
print(json.dumps(s))
