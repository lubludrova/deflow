"""Finish the successful smoke's report after the preflight import-path error."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code/package'))
from deflow_rebuttal.precision import apply_policy
v=root/'validation'
assert json.loads((v/'matched_config_gate.json').read_text())['passed']
for name in ('runner-tests','unit-tests','smoke','smoke-eval'):
    assert json.loads((v/(name+'-exit.json')).read_text())['exit_code']==0
for name,count in [('runner-tests',4),('unit-tests',27)]:
    log=(v/(name+'.log')).read_text()
    assert f'Ran {count} tests' in log and log.rstrip().endswith('OK'),log
assert not (v/'smoke/FAILED.json').exists()
done=json.loads((v/'smoke/DONE.json').read_text())
p=next(Path(x['path']) for x in done['checkpoints'] if Path(x['path']).name.startswith('ckpt_step_'))
state=torch.load(p,map_location='cpu',weights_only=False)
assert 4200<=state['global_step']<=4207
for name in ('actor','q1','q2'):
    assert all(bool(torch.isfinite(t).all()) for t in state[name].values())
assert bool(torch.isfinite(state['log_alpha']).all())
precision=json.loads((v/'smoke/runtime_precision.json').read_text())
for event in ('actor_forward','solver_entry','solver_result','density_result','backward'):
    assert precision['events'][event]['actual']['matmul_allow_tf32'] is False
    assert precision['events'][event]['dtypes']==['torch.float32']
events=EventAccumulator(str(p.parent),size_guidance={'scalars':0}).Reload()
for tag in ('loss/q_loss','loss/pi_loss','coeff/alpha'):
    assert tag in events.Tags()['scalars']
    assert all(math.isfinite(e.value) for e in events.Scalars(tag))
ev=json.loads((v/'smoke-eval/evaluation.json').read_text())
assert len(ev['episodes'])==10 and ev['eval_seed']==26152
assert ev['checkpoint_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
apply_policy('strict_fp32')
assert torch.cuda.device_count()==3
devices=[]
for i in range(3):
    with torch.cuda.device(i):
        x=torch.randn(4,3,device='cuda',requires_grad=True)
        (x@x.T).sum().backward()
        assert bool(torch.isfinite(x.grad).all())
        torch.cuda.synchronize()
        devices.append(torch.cuda.get_device_name(i))
assert torch.__version__=='2.7.1+cu128'
(v/'runtime-pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
receipt=dict(passed=True,torch=torch.__version__,python=sys.version,devices=devices,smoke_step=state['global_step'],smoke_elapsed_seconds=done['elapsed_seconds'],smoke_checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),smoke_eval_episodes=10,matched_config=True,unit_tests=27,supervisor_tests=4,preflight_note='Initial report failed at package import after successful smoke and evaluation. Finished report without scientific retry.',limitation='Finite-value/configuration gate, not all-sample convergence certification')
(v/'RELEASE_READY.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
