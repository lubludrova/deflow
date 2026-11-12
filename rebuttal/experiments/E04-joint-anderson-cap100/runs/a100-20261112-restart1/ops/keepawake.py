import json,time
from pathlib import Path
roots=['/Users/ruslanageev/dev/DEFlow/rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1', '/Users/ruslanageev/dev/DEFlow/rebuttal/experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1']
while True:
 states=[]
 for root in roots:
  path=Path(root)/'ops/monitor_state.json'
  states.append(json.loads(path.read_text()) if path.exists() else None)
 if all(s and all(j['state'] in ('EVALUATION_ACCEPTED','FAILED') for j in s['jobs']) for s in states): break
 time.sleep(30)
