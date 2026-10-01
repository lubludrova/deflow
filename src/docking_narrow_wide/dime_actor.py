"""DIME diffusion actor with an entropy lower-bound objective."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sac_anderson_flow as base
import sac_flow_parent as parent


LOG2PI = math.log(2.0 * math.pi)
PRIOR_STD = 2.5
DIME_DIFF_STEPS = 16
DT_INIT = 0.1
FRICTION_INIT = 1.0
COSINE_MIN = 0.001
COSINE_S = 0.008
COSINE_POW = 2
NUM_HID = 256
NUM_LAYERS = 3
OUTER_CLIP = 1e4
FINAL_W_INIT = 1e-8
ACTOR_GRAD_CLIP = 1.0


def inverse_softplus(x: float) -> float:
    return math.log(math.expm1(x))


def cosine_dt_schedule(T: int) -> torch.Tensor:


    k = torch.arange(T, dtype=torch.float32)
    t = (T - k) / T
    offset = 1.0 + COSINE_S
    return (1.0 - COSINE_MIN) * torch.cos(0.5 * math.pi * (offset - t) / offset) ** COSINE_POW + COSINE_MIN


def gauss_logp(x, mean, scale):

    return (-0.5 * ((x - mean) / scale) ** 2 - torch.log(scale) - 0.5 * LOG2PI).sum(-1)


def _zero_nans_and_clip(g):


    return torch.where(torch.isnan(g), torch.zeros_like(g), g).clamp(-ACTOR_GRAD_CLIP, ACTOR_GRAD_CLIP)


class dime_dis_actor(nn.Module):


    def __init__(self, obs_dim, act_dim, act_low, act_high, *,
                 denoising_steps: int = DIME_DIFF_STEPS, **kwargs):
        super().__init__()
        self.act_dim = act_dim
        self.T = denoising_steps
        self.register_buffer("dt_sched", cosine_dt_schedule(denoising_steps))

        self.raw_dt = nn.Parameter(torch.tensor([inverse_softplus(DT_INIT)]))
        self.raw_friction = nn.Parameter(torch.full((act_dim,), inverse_softplus(FRICTION_INIT)))


        self.timestep_phase = nn.Parameter(torch.zeros(1, NUM_HID))
        self.register_buffer("timestep_coeff", torch.linspace(0.1, 100.0, NUM_HID).unsqueeze(0))
        self.time_coder_state = nn.Sequential(
            nn.Linear(2 * NUM_HID, NUM_HID), nn.GELU(approximate="tanh"),
            nn.Linear(NUM_HID, NUM_HID),
        )
        layers = []
        d_in = obs_dim + act_dim + NUM_HID
        for _ in range(NUM_LAYERS):
            layers += [nn.Linear(d_in, NUM_HID), nn.GELU(approximate="tanh")]
            d_in = NUM_HID
        layers.append(nn.Linear(d_in, act_dim))
        nn.init.constant_(layers[-1].weight, FINAL_W_INIT)
        nn.init.zeros_(layers[-1].bias)
        self.state_time_net = nn.Sequential(*layers)

        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("action_scale", (act_high - act_low) / 2.0)
        self.register_buffer("action_bias", (act_high + act_low) / 2.0)


        for p in self.parameters():
            p.register_hook(_zero_nans_and_clip)

    def _time_features(self, k: int, B: int, device, dtype):

        t = torch.full((B, 1), float(k), device=device, dtype=dtype)
        ang = self.timestep_coeff * t + self.timestep_phase
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.time_coder_state(emb)

    def _score_net(self, x, obs, k):
        h = torch.cat([x, obs, self._time_features(k, x.shape[0], x.device, x.dtype)], dim=-1)
        return torch.clamp(self.state_time_net(h), -OUTER_CLIP, OUTER_CLIP)

    def _eta_scale(self):


        dt = F.softplus(self.raw_dt) * self.dt_sched
        sigma2 = 1.0 / F.softplus(self.raw_friction)
        eta = dt.unsqueeze(-1) * sigma2.unsqueeze(0)
        return eta, torch.sqrt(2.0 * eta)

    def forward_with_logprob(self, obs):
        B = obs.shape[0]
        device, dtype = obs.device, obs.dtype
        eta, scale = self._eta_scale()


        x = (torch.randn(B, self.act_dim, device=device, dtype=dtype) * PRIOR_STD).detach()

        terminal = (-0.5 * (x / PRIOR_STD) ** 2 - math.log(PRIOR_STD) - 0.5 * LOG2PI).sum(-1)

        log_w = torch.zeros(B, device=device, dtype=dtype)
        for k in range(self.T):


            drift = -x / PRIOR_STD ** 2
            fwd_mean = x + eta[k] * (drift + self._score_net(x, obs, k))
            x_new = fwd_mean + scale[k] * torch.randn_like(x)
            bwd_mean = x_new + eta[k] * (-x_new / PRIOR_STD ** 2)
            log_w = log_w + gauss_logp(x, bwd_mean, scale[k]) - gauss_logp(x_new, fwd_mean, scale[k])
            x = x_new


        tanh_ld = (2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))).sum(-1)
        run = -(log_w + tanh_ld)
        sto = torch.zeros_like(run)
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias


        logp = run + sto + terminal
        return action, logp

    def forward(self, obs):
        return self.forward_with_logprob(obs)[0]

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        if not deterministic:
            return self.forward(obs)


        B = obs.shape[0]
        eta, _ = self._eta_scale()
        x = torch.zeros(B, self.act_dim, device=obs.device, dtype=obs.dtype)
        for k in range(self.T):
            x = x + eta[k] * (-x / PRIOR_STD ** 2 + self._score_net(x, obs, k))
        return torch.tanh(x) * self.action_scale + self.action_bias


base.deq_multistep_flow_actor = dime_dis_actor
base.critic = parent.parent_critic
