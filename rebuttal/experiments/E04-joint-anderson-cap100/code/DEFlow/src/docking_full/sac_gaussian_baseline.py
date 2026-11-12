"""Tanh-Gaussian actor for the SAC baseline."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sac_anderson_flow as base

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


class gaussian_actor(nn.Module):

    def __init__(self, obs_dim, act_dim, act_low, act_high, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("action_scale", (act_high - act_low) / 2.0)
        self.register_buffer("action_bias", (act_high + act_low) / 2.0)

    def forward_with_logprob(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(-1)

    def forward(self, x):
        return self.forward_with_logprob(x)[0]

    @torch.no_grad()
    def act(self, x, deterministic: bool = False):
        if not deterministic:
            return self.forward(x)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.tanh(self.fc_mean(h)) * self.action_scale + self.action_bias


base.deq_multistep_flow_actor = gaussian_actor

if __name__ == "__main__":
    args = base.Args()
    args.exp_name = "sac_gaussian_baseline"
    base.main(args)
