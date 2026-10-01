"""Train Gaussian SAC with a matched twin-Q critic."""

import os
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sac_anderson_flow as m
import sac_gaussian_baseline as G
import sac_flow_parent as P

m.deq_multistep_flow_actor = G.gaussian_actor
m.critic = P.parent_critic

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("environment")
env_id = parser.parse_args().environment
a = m.Args()
a.exp_name = "sac"
a.env_id = env_id
a.seed = 1
a.num_envs = 8
a.total_timesteps = 1_000_000
a.target_entropy_scale = 1.0

a.batch_size = 512
a.learning_starts = 50_000
a.policy_frequency = 1
a.eval_interval = 25_000
a.save_model = True
m.main(a)
