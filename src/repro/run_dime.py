"""Train DIME with a matched twin-Q critic."""

import math
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dime_actor
import sac_anderson_flow as base


print(f"probe: base.deq_multistep_flow_actor={base.deq_multistep_flow_actor.__module__}."
      f"{base.deq_multistep_flow_actor.__name__}  "
      f"base.critic={base.critic.__module__}.{base.critic.__name__}")

_eval_orig = base.eval_policy
def _eval_probe(eval_env, actor_net, device, num_envs, eval_seed=None):
    print(f"probe: type(actor).__module__={type(actor_net).__module__}  "
          f"type(actor).__name__={type(actor_net).__name__}")
    base.eval_policy = _eval_orig
    return _eval_orig(eval_env, actor_net, device, num_envs, eval_seed=eval_seed)
base.eval_policy = _eval_probe

_soft_orig = base.soft_update
def _soft_probe(target, source, tau):
    print(f"probe: type(critic).__module__={type(source).__module__}  "
          f"type(critic).__name__={type(source).__name__}")
    base.soft_update = _soft_orig
    return _soft_orig(target, source, tau)
base.soft_update = _soft_probe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("environment")
env_id = parser.parse_args().environment
a = base.Args()
a.exp_name = "dime"
a.env_id = env_id
a.seed = 1
a.num_envs = 8
a.total_timesteps = 1_000_000
a.learning_starts = 50_000
a.batch_size = 512
a.policy_frequency = 1
a.utd_ratio = 1.0
a.eval_interval = 25_000
a.save_model = True

a.alpha_init = 1.0


a.target_entropy_scale = 4.0 - (0.5 * math.log(2 * math.pi * math.e) + math.log(dime_actor.PRIOR_STD))
a.denoising_steps = 16


a.alpha_min = 0.0

a.logp_clip = 0.0
a.grad_clip = 0.0
base.main(a)
