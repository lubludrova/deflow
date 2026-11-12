"""Check completed preflight artifacts before freezing production launch."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/package'))
from deflow_rebuttal.precision import apply_policy, source_hashes

apply_policy('strict_fp32')
assert json.loads((ROOT/'validation/matched_config_gate.json').read_text())['passed']
assert json.loads((ROOT/'validation/fixed-capsules-v3/report.json').read_text())['implementation_gate_pass']
for filename, count in [('unit-tests-v2.log',27),('runner-tests.log',4)]:
    log=(ROOT/'validation'/filename).read_text()
    assert f'Ran {count} tests' in log and log.rstrip().endswith('OK')
pilot=ROOT/'validation/pilot'
assert not (pilot/'FAILED.json').exists()
done=json.loads((pilot/'DONE.json').read_text())
checkpoint=next(Path(c['path']) for c in done['checkpoints'] if Path(c['path']).name.startswith('ckpt_step_'))
payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert 8000<=payload['global_step']<=8007
for name in ('actor','q1','q2'):
    assert all(bool(torch.isfinite(t).all()) for t in payload[name].values()),name
assert bool(torch.isfinite(payload['log_alpha']).all())
pilot_manifest=json.loads((pilot/'manifest.json').read_text())
assert pilot_manifest['package_sha256']==source_hashes()
runtime=json.loads((pilot/'runtime_precision.json').read_text())
for event in ('actor_forward','solver_entry','solver_result','density_result','backward'):
    assert runtime['events'][event]['actual']['matmul_allow_tf32'] is False
    assert runtime['events'][event]['dtypes']==['torch.float32']
events=EventAccumulator(str(checkpoint.parent),size_guidance={'scalars':0}).Reload()
for tag in ('loss/q_loss','loss/pi_loss','coeff/alpha'):
    assert tag in events.Tags()['scalars']
    assert all(__import__('math').isfinite(e.value) for e in events.Scalars(tag))
evaluation=json.loads((ROOT/'validation/smoke-eval/evaluation.json').read_text())
assert len(evaluation['episodes'])==10 and evaluation['eval_seed']==26152
devices=[]
for device in range(3):
    with torch.cuda.device(device):
        value=torch.randn(4,3,device='cuda',requires_grad=True)
        (value@value.T).sum().backward()
        assert torch.isfinite(value.grad).all()
        torch.cuda.synchronize()
        devices.append(torch.cuda.get_device_name(device))
receipt=dict(passed=True,unit_tests=27,runner_tests=4,pilot_step=payload['global_step'],
             pilot_elapsed_seconds=done['elapsed_seconds'],pilot_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             smoke_evaluation_episodes=10,devices=devices,torch=torch.__version__,
             package_sha256=source_hashes(),limitations='Implementation/finite-value gate, not all-sample convergence certification')
(ROOT/'validation/RELEASE_READY.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
