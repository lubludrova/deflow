"""One bounded full1024 synthetic end-to-end CPU check, not scientific evidence."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import torch
from torch import nn

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / 'package'))
spec = importlib.util.spec_from_file_location('numerical_preflight', here / 'numerical_preflight.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from deflow_rebuttal import audit, precision

torch.set_num_threads(1)
torch.manual_seed(41)
precision.apply_policy('strict_fp32')
engine = audit.load_engine(here.parent / 'DEFlow/src/docking_full/sac_anderson_flow.py')
kwargs = dict(obs_dim=3, act_dim=2, act_low=-torch.ones(2), act_high=torch.ones(2),
              hidden_dim=8, time_emb_dim=4, denoising_steps=4, exact_backward=True)
actor = engine.deq_multistep_flow_actor(**kwargs)
critics = [nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1)) for _ in range(2)]
payload = dict(actor=actor.state_dict(), actor_kwargs=kwargs, obs=torch.randn(1024, 3),
               z=torch.randn(1024, 2), metadata=dict(role='training/actor_grad', global_step=123),
               loss_config=dict(alpha=.2, logp_lower=-100., logp_upper=100., effective_lam_jac=0.))
payload.update({name: {f'net.{k}': v for k, v in net.state_dict().items()}
                for name, net in zip(('q1', 'q2'), critics)})
# Keep a new test artifact directory, never alter any scientific capsule.
directory = Path(tempfile.mkdtemp(prefix='synthetic_cpu_', dir=here))
path = directory / 'capsule.pt'
torch.save(payload, path)
record = dict(path=str(path), sha256=module.digest(path), actual_step=123,
              source='synthetic_cpu_fixture', rule='Fixed RNG seed41, all1024 synthetic rows')
result = module.check_case(record, directory, engine, 'cpu', audit, precision)
print(directory)
print({key: result[key] for key in ('pass_all', 'blockers', 'reference_valid_rows', 'fd_pass', 'parity_pass')})
raise SystemExit(0 if result['pass_all'] else 1)
