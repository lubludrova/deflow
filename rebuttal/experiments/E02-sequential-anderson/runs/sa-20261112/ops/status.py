"""Read current remote identities, semantic progress, and evaluation receipts."""
import json
from pathlib import Path
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
profile=json.loads((HERE/'profile.json').read_text())
program=r'''
import hashlib,json,re,sys,time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path(sys.argv[1])
manifest=json.loads((root/'ops/manifest.json').read_text())
rows=[]
for job in manifest['jobs']:
 run_id=job['run_id']
 attempt=root/'attempts'/run_id/'0000'
 row=dict(run_id=run_id,gpu=job['gpu'],state='PLANNED',step=None)
 if attempt.exists():
  row['state']='CLAIMED'
  live=attempt/'live.json'
  if live.exists():
   data=json.loads(live.read_text())
   row.update(data)
   row['gpu']=job['gpu']
   ident=data['identity']
   proc=Path('/proc')/str(ident['pid'])
   same=False
   if proc.exists():
    ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    same=ticks==ident['start_ticks'] and hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()==ident['command_sha256']
   row['process_identity_matches']=same
   row['state']=data['phase'].upper() if same else 'AWAITING_TERMINAL_RECEIPT'
  for marker,state in [('TRAINING_ACCEPTED.json','TRAINING_ACCEPTED'),('EVALUATION_ACCEPTED.json','EVALUATION_ACCEPTED'),('FAILED.json','FAILED')]:
   p=attempt/marker
   if p.exists():
    row[marker]=json.loads(p.read_text())
    if state!='TRAINING_ACCEPTED' or row['state']!='EVALUATION': row['state']=state
  log=attempt/'training.log'
  if log.exists():
   with log.open('rb') as f:
    f.seek(max(0,log.stat().st_size-131072))
    tail=f.read().decode(errors='replace')
   steps=re.findall(r'\bstep\s*[=:]\s*(\d+)',tail)
   if steps: row['step']=int(steps[-1])
  events=list((root/'results'/run_id/'runs').glob('*/events.out.tfevents.*'))
  if events:
   acc=EventAccumulator(str(events[0].parent),size_guidance={'scalars':0}).Reload()
   for tag,key in [('charts/avg_episode_reward','train_return'),('charts/eval_return_det','eval_return')]:
    if tag in acc.Tags()['scalars']:
     event=acc.Scalars(tag)[-1]
     row[key]=dict(value=event.value,step=event.step,time=event.wall_time)
 rows.append(row)
print(json.dumps(dict(timestamp=time.time(),jobs=rows)))
'''
server=profile['server']
command=["ssh","-S",profile['control_socket'],"-o","BatchMode=yes","-o","ConnectTimeout=20",
         "-p",str(server['port']),server['user']+'@'+server['host'],
         shlex.quote(profile['remote_root']+'/venv/bin/python')+' -c '+shlex.quote(program)+' '+shlex.quote(profile['remote_root'])]
result=json.loads(subprocess.check_output(command,text=True))
previous_path=HERE/'last_status.json'
if previous_path.exists():
    previous=json.loads(previous_path.read_text())
    elapsed=result['timestamp']-previous['timestamp']
    old={x['run_id']:x for x in previous['jobs']}
    for row in result['jobs']:
        prior=old.get(row['run_id'],{})
        if elapsed>0 and row.get('step') is not None and prior.get('step') is not None:
            delta=row['step']-prior['step']
            if delta>0:
                row['delta_sps']=delta/elapsed
                row['delta_window_seconds']=elapsed
                row['training_eta_seconds']=max(0,200000-row['step'])/row['delta_sps']
previous_path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
