"""Observe only. Persist progress, recent speed, sync freshness and alerts."""
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ops=Path(__file__).resolve().parent
lock=(ops/'monitor.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    result=subprocess.run([sys.executable,str(ops/'status.py')],capture_output=True,text=True,timeout=45,check=True)
    status=json.loads(result.stdout)
    ackpath=ops/'SYNC_ACK.json'
    ack=json.loads(ackpath.read_text()) if ackpath.exists() else {}
    status['sync_age_seconds']=time.time()-ack.get('timestamp',0)
    status['verified_artifacts']=len(ack.get('critical_sha256',{}))
    status['alerts']=[]
    if status['sync_age_seconds']>600:status['alerts'].append('SYNC_STALE')
    for row in status['jobs']:
        if row['state']=='FAILED':status['alerts'].append(row['run_id']+':FAILED')
        if row['state']=='AWAITING_TERMINAL_RECEIPT':status['alerts'].append(row['run_id']+':IDENTITY_OR_TERMINAL_PENDING')
        if row['state']=='TRAINING' and time.time()-row.get('log_modified',0)>600:status['alerts'].append(row['run_id']+':PROGRESS_STALE')
    status['all_evaluated']=all(j['state']=='EVALUATION_ACCEPTED' for j in status['jobs'])
    temp=ops/'monitor_state.tmp'
    temp.write_text(json.dumps(status,indent=2)+'\n')
    temp.replace(ops/'monitor_state.json')
    with (ops/'monitor_history.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    print(json.dumps(dict(time=status['timestamp'],states=[(j['run_id'],j['state'],j.get('step'),j.get('delta_sps')) for j in status['jobs']],sync_age=status['sync_age_seconds'],alerts=status['alerts'])),flush=True)
except Exception as e:
    with (ops/'monitor_errors.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),error=str(e)))+'\n')
    raise
