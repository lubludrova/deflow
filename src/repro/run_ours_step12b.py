"""Train DEFlow with the selected MuJoCo configuration."""

import os
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sac_anderson_flow as m

_REAL = m.deq_multistep_flow_actor
import sac_flow_parent as P


class tuned_actor(_REAL):
    def __init__(self, *a, **kw):
        kw["hidden_dim"] = 256
        kw["max_iter_fwd"] = 25
        kw["tol_fwd"] = 3e-4
        kw["exact_backward"] = True
        super().__init__(*a, **kw)


m.deq_multistep_flow_actor = tuned_actor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("environment")
env_id = parser.parse_args().environment
a = m.Args()
a.exp_name = "deflow"
a.env_id = env_id
a.seed = 1
a.num_envs = 8
a.total_timesteps = 1_000_000
a.u_scale = 4.0

a.gate_bias = 1.0
a.lam_jac = 1e-2
a.jac_sigma_target = 0.9
a.jac_warmup = 100_000

a.ent_track = True
a.ent_margin = 0.7
a.ent_floor_scale = 0.7
a.alpha_leak = 1e-2
a.alpha_prior = 0.05
a.lp_winsor = 0.05
a.logp_clip = 15.0
a.grad_clip = 1.0
a.alpha_min = 0.02


a.batch_size = 512
a.learning_starts = 50_000
a.policy_frequency = 1
a.eval_interval = 25_000
a.save_model = True
m.main(a)
