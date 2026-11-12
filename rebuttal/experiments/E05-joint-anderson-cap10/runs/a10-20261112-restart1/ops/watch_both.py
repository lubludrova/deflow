"""Read-only wait for both existing durable campaign monitors."""
import json,time
from pathlib import Path
base=Path(__file__).resolve().parents[4]
paths=[base/'E04-joint-anderson-cap100/runs/a100-20261112-restart1/ops',base/'E05-joint-anderson-cap10/runs/a10-20261112-restart1/ops']
seen=set()
last_report=0
while True:
    now=time.time()
    statuses=[json.loads((p/'monitor_state.json').read_text()) for p in paths]
    changed=False
    for p,s in zip(paths,statuses):
        alerts=s['alerts'][:]
        if now-s['timestamp']>180:alerts.append('MONITOR_STALE')
        for alert in alerts:
            key=(str(p),alert)
            if key not in seen:
                print(json.dumps(dict(event='ALERT',time=now,campaign=p.parts[-4],alert=alert)),flush=True)
                seen.add(key)
        for j in s['jobs']:
            key=(j['run_id'],j['state'])
            if j['state'] in ('EVALUATION','EVALUATION_ACCEPTED','TRAINING_ACCEPTED','FAILED') and key not in seen:
                changed=True
                seen.add(key)
    terminal=all(j['state'] in ('EVALUATION_ACCEPTED','FAILED') for s in statuses for j in s['jobs'])
    if changed or terminal or now-last_report>=600:
        print(json.dumps(dict(event='TERMINAL' if terminal else 'STATUS',time=now,sync_ages=[s['sync_age_seconds'] for s in statuses],jobs=[{k:j.get(k) for k in ('run_id','state','step','train_return','eval_return','delta_sps','elapsed_seconds','training_eta_seconds')} for s in statuses for j in s['jobs']])),flush=True)
        last_report=now
    if terminal:break
    time.sleep(30)
