"""Verify compact mirror against remote receipt and recompute episode metrics."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

base=Path(__file__).resolve().parents[1]
run=base
mirror=run/'mirror'
verification=json.loads((base/'analysis/remote-verification.json').read_text())
ack=json.loads((run/'ops/FINAL_SYNC_ACK.json').read_text())
assert verification['passed']
assert verification['manifest_sha256']==ack['manifest_sha256']
for relative,expected in verification['critical_sha256'].items():
    p=mirror/relative
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,relative
    assert ack['critical_sha256'][relative]==expected,relative
rows=[]
for remote in verification['rows']:
    ev=json.loads((mirror/'evaluations'/remote['run_id']/'evaluation.json').read_text())
    episodes=ev['episodes']
    assert len(episodes)==1000 and ev['eval_seed']==26152
    assert [e['episode'] for e in episodes]==list(range(1000))
    assert all(e['success'] in (0,1) and math.isfinite(e['return']) for e in episodes)
    p=[sum(e['success']==1 and e['goal']==i for e in episodes)/1000 for i in range(4)]
    success=sum(p)
    u4=sum(1-(1-v)**4 for v in p)/4
    ret=statistics.mean(e['return'] for e in episodes)
    for key,value in [('total_success',success),('u4',u4),('mean_return',ret)]:
        assert math.isclose(value,ev['summary'][key],rel_tol=1e-12,abs_tol=1e-12),(key,value)
    rows.append(dict(seed=remote['seed'],steps=remote['step'],u4=u4,success_percent=100*success,
        mean_return=ret,training_sps=remote['training_sps'],training_elapsed_seconds=remote['training_elapsed_seconds']))
metrics={key:dict(mean=statistics.mean(r[key] for r in rows),sample_sd=statistics.stdev(r[key] for r in rows)) for key in ('u4','success_percent','mean_return','training_sps')}
with (base/'analysis/per_seed.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
result=dict(verified_at=datetime.now(timezone.utc).isoformat(),campaign='E05 joint Anderson tol1e-5 cap10',seeds=rows,mean_and_sample_sd=metrics,remote_and_local_hashes_verified=len(verification['critical_sha256']),episodes_per_seed=1000,eval_seed=26152,retries=0,resume=False,sps_definition=verification['sps_definition'])
(base/'analysis/metrics.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# E05 — joint Anderson tolerance 1e-5, cap 10','',
       'All three fresh seeds completed training and their 1000-episode stochastic evaluations with exit 0. Each final checkpoint and all eight periodic/final checkpoints per seed were loaded remotely and checked for finite actor/critic/temperature values. Local SHA-256 matches the remote verification receipt and final sync acknowledgment.',
       '', '| Seed | Real transitions | U4 | Success (%) | Return | Training SPS |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
    lines.append(f"| {r['seed']} | {r['steps']} | {r['u4']:.6f} | {r['success_percent']:.2f} | {r['mean_return']:.5f} | {r['training_sps']:.3f} |")
values=[f"{metrics[k]['mean']:.4f} ± {metrics[k]['sample_sd']:.4f}" for k in ('u4','success_percent','mean_return','training_sps')]
lines.append('| Mean ± sample SD | — | '+' | '.join(values)+' |')
lines.extend(['', 'Sample SD uses n−1 across three training seeds. Evaluation seed is 26152. U4, success and return were independently recomputed from all 1000 episode records per seed. No seed or episode was filtered.', '',
    'Training SPS = actual real transitions / DONE.elapsed_seconds. This includes unchanged diagnostics, checkpoint writing and intermediate evaluation, and excludes the final stochastic evaluation. It is end-to-end training throughput, not isolated solver performance. All seeds used RTX5090 with E02 PyTorch 2.7.1+cu128 and strict FP32. Each seed trained concurrently with its E04 cap100 peer. Shared-GPU timing is recorded in launch and exit receipts. Final E05 evaluations waited for E04 completion and GPU availability. Historical E01 shared RTX4090 timings are not a controlled speed comparator.', '',
    'Protocol: MultiGoalDockTight-v0, 200000 requested transitions, num_envs=8, batch=1024, learning_starts=4000, utd_ratio=0.5. Native joint Anderson tolerance1e-5/cap10, m5/lam1e-2/beta0.7 and unchanged shared density/backward. This condition is separate from sequential Anderson and tighter tolerance1e-6/cap100.', '',
    'Evidence: remote-verification.json, metrics.json, per_seed.csv, ../FROZEN.json, LAUNCH_VERIFIED.json, ops/manifest.json, ops/COMMANDS.txt and ops/FINAL_SYNC_ACK.json. Full raw solver traces/capsules stay on the server. Nothing was deleted, retried or resumed.', '',
    'Operational note: preflight configuration, smoke training, smoke evaluation and shared-placement gates passed. Two additional tests checked peer identity admission and the final ACK guard. No scientific retries or resume.'])
(base/'analysis/REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
