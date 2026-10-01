"""Train QSM with a matched twin-Q critic."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qsm_actor as m

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("environment")
env_id = parser.parse_args().environment
a = m.QSMArgs()
a.exp_name = "qsm"
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

m.main(a)
