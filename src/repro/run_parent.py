"""Train SAC-Flow with a matched twin-Q critic."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sac_flow_parent
import sac_anderson_flow as base

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("environment")
env_id = parser.parse_args().environment
a = base.Args()
a.exp_name = "sacflow"
a.env_id = env_id
a.seed = 1
a.num_envs = 8
a.total_timesteps = 1_000_000
a.learning_starts = 50_000
a.batch_size = 512
a.alpha_init = 0.2
a.target_entropy_scale = 0.0
a.policy_frequency = 1
a.utd_ratio = 1.0
a.eval_interval = 25_000
a.save_model = True
base.main(a)
