"""Read-only final validation of checkpoints, sources, exits and evaluation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import torch

root=Path(sys.argv[1])
m=json.loads((root/'ops/manifest.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for rel,expected in m['frozen_files'].items():
    assert sha(root/rel)==expected,rel
rows=[]
critical={}
for job in m['jobs']:
    rid=job['run_id']
    attempt=root/'attempts'/rid/'0000'
    assert not (attempt/'FAILED.json').exists()
    exits={phase:json.loads((attempt/(phase+'_exit.json')).read_text()) for phase in ('training','evaluation')}
    assert all(x['exit_code']==0 for x in exits.values())
    result=root/'results'/rid
    done=json.loads((result/'DONE.json').read_text())
    manifest=json.loads((result/'manifest.json').read_text())
    assert manifest['solver']=='anderson' and manifest['anderson']==dict(tol=1e-5,cap=100)
    assert manifest['precision_policy']=='strict_fp32'
    precision=json.loads((result/'runtime_precision.json').read_text())
    for event in ('actor_forward','solver_entry','solver_result','density_result','backward'):
        assert precision['events'][event]['dtypes']==['torch.float32']
        assert precision['events'][event]['actual']['matmul_allow_tf32'] is False
    checkpoints=sorted(result.glob('runs/*/ckpt_step_*.pt'))
    steps=[]
    for p in checkpoints:
        state=torch.load(p,map_location='cpu',weights_only=False)
        step=state['global_step']
        steps.append(step)
        assert state['args']['seed']==job['seed']
        assert state['args']['rebuttal_solver']=='anderson'
        assert state['args']['rebuttal_anderson']==dict(tol=1e-5,cap=100)
        assert state['args']['env_id']=='MultiGoalDockTight-v0'
        for name in ('actor','q1','q2'):
            assert all(bool(torch.isfinite(t).all()) for t in state[name].values())
        assert bool(torch.isfinite(state['log_alpha']).all())
        critical[str(p.relative_to(root))]=sha(p)
    assert len(steps)==8,steps
    assert all(target<=actual<=target+7*i for i,(target,actual) in enumerate(zip(range(25000,200000,25000),steps[:7]),1)),steps
    assert 200000<=steps[-1]<=200007,steps
    accepted=json.loads((attempt/'TRAINING_ACCEPTED.json').read_text())
    final=root/accepted['checkpoint']
    assert sha(final)==accepted['sha256']
    evaluation=root/'evaluations'/rid/'evaluation.json'
    ev=json.loads(evaluation.read_text())
    assert len(ev['episodes'])==1000 and ev['eval_seed']==26152
    assert ev['checkpoint_sha256']==sha(final) and ev['solver']=='anderson'
    critical[str(evaluation.relative_to(root))]=sha(evaluation)
    assert sha(evaluation)==json.loads((attempt/'EVALUATION_ACCEPTED.json').read_text())['sha256']
    assert all(math.isfinite(v) for v in ev['summary'].values() if isinstance(v,(int,float)))
    rows.append(dict(seed=job['seed'],run_id=rid,gpu=job['gpu'],step=steps[-1],checkpoint_steps=steps,
        summary=ev['summary'],training_elapsed_seconds=done['elapsed_seconds'],
        training_sps=steps[-1]/done['elapsed_seconds'],training_supervisor_elapsed_seconds=exits['training']['elapsed_seconds'],
        evaluation_elapsed_seconds=exits['evaluation']['elapsed_seconds'],
        training_exit=0,evaluation_exit=0,checkpoint_sha256=sha(final),evaluation_sha256=sha(evaluation)))
print(json.dumps(dict(passed=True,manifest_sha256=sha(root/'ops/manifest.json'),rows=rows,critical_sha256=critical,
    sps_definition='actual real transitions / DONE elapsed_seconds, includes diagnostics and intermediate eval, excludes final stochastic eval'),indent=2))
