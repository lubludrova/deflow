"""Validate the exact A100 condition without changing production sources."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import runner
m=json.loads((ROOT/'ops/manifest.json').read_text())
v=ROOT/'validation'
v.mkdir(exist_ok=True)
def save(name,data):
    (v/name).write_text(json.dumps(data,indent=2)+'\n')
def execute(name,argv,env):
    start=time.time()
    with (v/(name+'.log')).open('x') as log:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    save(name+'-exit.json',dict(argv=argv,exit_code=result.returncode,elapsed_seconds=time.time()-start))
    assert result.returncode==0,name

env=os.environ.copy()
env.update(m['environment'])
env['CUDA_VISIBLE_DEVICES']=m['gpu_by_slot']['0']
samples=[]
for i in range(7):
    s=runner.gpu_snapshot()
    assert not s['processes'],s
    for row in s['gpus']:
        slot,uuid,util,used,free=[x.strip() for x in row.split(',')]
        assert m['gpu_by_slot'][slot]==uuid and int(util)<10 and int(free)>12000
    samples.append(s)
    if i<6:time.sleep(10)
save('preflight-gpu-window.json',samples)
reference=Path('/home/featurize/deflow-sequential-anderson-20261112/results/sequential_anderson_s1/manifest.json')
baseline=json.loads(reference.read_text())
plans=[]
allowed={'seed','run_name','exp_name','rebuttal_solver','rebuttal_anderson','rebuttal_actor_kwargs_sha256'}
for job in m['jobs']:
    argv=job['train_argv'][:-1]+['--plan']
    plan=json.loads(subprocess.check_output(argv,env=env,text=True))
    diffs={k:dict(baseline=baseline['configuration'].get(k),actual=plan['configuration'].get(k)) for k in baseline['configuration'].keys()|plan['configuration'].keys() if baseline['configuration'].get(k)!=plan['configuration'].get(k)}
    assert set(diffs)<=allowed,diffs
    c=plan['configuration']
    assert (c['env_id'],c['total_timesteps'],c['num_envs'],c['batch_size'],c['learning_starts'],c['utd_ratio'],c['ckpt_interval'])==('MultiGoalDockTight-v0',200000,8,1024,4000,0.5,25000)
    assert plan['solver']=='anderson' and plan['anderson']==dict(tol=1e-5,cap=100)
    assert plan['source_sha256']==baseline['source_sha256']
    assert plan['package_sha256']==baseline['package_sha256']
    save('plan-s'+str(job['seed'])+'.json',plan)
    plans.append(dict(seed=job['seed'],differences=diffs))
save('matched_config_gate.json',dict(passed=True,baseline=str(reference),baseline_sha256=runner.digest(reference),plans=plans))
execute('runner-tests',[sys.executable,'-m','unittest','discover','-s',str(ROOT/'ops'),'-p','test_runner.py'],env)
execute('unit-tests',[sys.executable,'-m','unittest','discover','-s',str(ROOT/'code/package/deflow_rebuttal'),'-t',str(ROOT/'code/package')],env)
argv=m['jobs'][0]['train_argv'].copy()
argv[argv.index('--output')+1]=str(v/'smoke')
argv[argv.index('--steps')+1]='4200'
execute('smoke',argv,env)
done=json.loads((v/'smoke/DONE.json').read_text())
checkpoint=next(Path(x['path']) for x in done['checkpoints'] if Path(x['path']).name.startswith('ckpt_step_'))
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert 4200<=payload['global_step']<=4207
for name in ('actor','q1','q2'):
    assert all(bool(torch.isfinite(t).all()) for t in payload[name].values())
assert bool(torch.isfinite(payload['log_alpha']).all())
precision=json.loads((v/'smoke/runtime_precision.json').read_text())
for event in ('actor_forward','solver_entry','solver_result','density_result','backward'):
    assert precision['events'][event]['actual']['matmul_allow_tf32'] is False
    assert precision['events'][event]['dtypes']==['torch.float32']
events=EventAccumulator(str(checkpoint.parent),size_guidance={'scalars':0}).Reload()
for tag in ('loss/q_loss','loss/pi_loss','coeff/alpha'):
    assert tag in events.Tags()['scalars'],tag
    assert all(math.isfinite(e.value) for e in events.Scalars(tag))
argv=m['jobs'][0]['eval_argv'].copy()
argv[argv.index('--run')+1]=str(v/'smoke')
argv[argv.index('--checkpoint')+1]=str(checkpoint)
argv[argv.index('--output')+1]=str(v/'smoke-eval')
argv[argv.index('--episodes')+1]='10'
execute('smoke-eval',argv,env)
evaldata=json.loads((v/'smoke-eval/evaluation.json').read_text())
assert len(evaldata['episodes'])==10 and evaldata['eval_seed']==26152
assert evaldata['solver']=='anderson' and evaldata['checkpoint_sha256']==runner.digest(checkpoint)
# This supervisor process has no CUDA_VISIBLE_DEVICES restriction.
from deflow_rebuttal.precision import apply_policy
apply_policy('strict_fp32')
devices=[]
assert torch.cuda.device_count()==3
for i in range(3):
    with torch.cuda.device(i):
        x=torch.randn(4,3,device='cuda',requires_grad=True)
        (x@x.T).sum().backward()
        assert bool(torch.isfinite(x.grad).all())
        torch.cuda.synchronize()
        devices.append(torch.cuda.get_device_name(i))
assert torch.__version__=='2.7.1+cu128'
(v/'runtime-pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
save('RELEASE_READY.json',dict(passed=True,torch=torch.__version__,python=sys.version,devices=devices,smoke_step=payload['global_step'],smoke_elapsed_seconds=done['elapsed_seconds'],smoke_checkpoint_sha256=runner.digest(checkpoint),smoke_eval_episodes=10,matched_config=True,precision_events=list(precision['events']),limitation='Finite-value and configuration smoke check, not convergence certification'))
print(json.dumps(json.loads((v/'RELEASE_READY.json').read_text()),indent=2))
