"""SAC-Flow actor and critic for matched policy comparisons."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sac_anderson_flow as base

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
PARENT_HIDDEN = 256
PARENT_CRITIC_WIDTH = 512
PARENT_CRITIC_DEPTH = 4


class parent_flow_actor(nn.Module):


    def __init__(self, obs_dim, act_dim, act_low, act_high, *,
                 denoising_steps: int = 4, time_emb_dim: int = 32, **kwargs):
        super().__init__()
        self.act_dim = act_dim
        self.T = denoising_steps
        self.dt = 1.0 / denoising_steps

        self.register_buffer(
            "time_steps",
            torch.linspace(0.0, 1.0 - self.dt, denoising_steps).reshape(-1, 1),
        )


        self.time_mlp = nn.Sequential(
            base.sinusoidal_pos_emb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        in_dim = obs_dim + act_dim + time_emb_dim

        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, PARENT_HIDDEN), nn.SiLU(),
            nn.Linear(PARENT_HIDDEN, act_dim),
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(in_dim, PARENT_HIDDEN), nn.SiLU(),
            nn.Linear(PARENT_HIDDEN, act_dim),
        )

        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 5.0)


        self.fc_logstd = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, act_dim),
        )

        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("action_scale", (act_high - act_low) / 2.0)
        self.register_buffer("action_bias", (act_high + act_low) / 2.0)

    def _flow_field(self, t, x, obs):
        temb = self.time_mlp(t)
        net_in = torch.cat([obs, x, temb], dim=-1)
        z = torch.sigmoid(self.gate_net(net_in))
        h_tilde = self.candidate_net(net_in)
        return z * (h_tilde - x)

    def _state_std(self, obs):
        log_std = torch.tanh(self.fc_logstd(obs))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return log_std.exp()

    def forward_with_logprob(self, obs):
        B = obs.shape[0]
        dev, dt_ = obs.device, obs.dtype
        LOG2PI = math.log(2.0 * math.pi)

        x = torch.randn(B, self.act_dim, device=dev, dtype=dt_)
        logp = (-0.5 * x.pow(2) - 0.5 * LOG2PI).sum(-1)
        std = self._state_std(obs)

        for step in range(self.T):
            t = self.time_steps[step].to(dev).expand(B, 1)
            u = self._flow_field(t, x, obs)
            mean_next = x + u * self.dt
            x = mean_next + std * torch.randn_like(mean_next)
            logp = logp + (-0.5 * ((x - mean_next) / std).pow(2)
                           - 0.5 * LOG2PI - torch.log(std)).sum(-1)

        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        logp = logp - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6).sum(-1)
        return action, logp

    def forward(self, obs):
        return self.forward_with_logprob(obs)[0]

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        if not deterministic:
            return self.forward(obs)


        B = obs.shape[0]
        x = torch.zeros(B, self.act_dim, device=obs.device, dtype=obs.dtype)
        for step in range(self.T):
            t = self.time_steps[step].to(obs.device).expand(B, 1)
            x = x + self._flow_field(t, x, obs) * self.dt
        return torch.tanh(x) * self.action_scale + self.action_bias


class parent_critic(nn.Module):


    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        layers = []
        d = obs_dim + act_dim
        for _ in range(PARENT_CRITIC_DEPTH):
            layers.append(nn.Linear(d, PARENT_CRITIC_WIDTH))
            layers.append(nn.ReLU())
            d = PARENT_CRITIC_WIDTH
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


base.deq_multistep_flow_actor = parent_flow_actor
base.critic = parent_critic


def _parent_args():
    args = base.Args()
    args.exp_name = "sac_flow_parent"

    args.batch_size = 512
    args.learning_starts = 50_000
    args.alpha_init = 0.2
    args.target_entropy_scale = 0.0
    args.policy_frequency = 1
    args.utd_ratio = 1.0

    args.alpha_min = 0.0

    args.logp_clip = 0.0
    args.grad_clip = 0.0
    args.denoising_steps = 4
    return args


if __name__ == "__main__":
    base.main(_parent_args())
