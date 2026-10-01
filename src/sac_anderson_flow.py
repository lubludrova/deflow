"""SAC training with an implicit flow policy and equilibrium differentiation."""

import os
import random
import time
import uuid
import copy
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math
from torch.func import vmap, jacrev

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    run_name: str = ""
    seed: int = 1
    num_envs: int = 8
    capture_video: bool = False
    save_model: bool = False
    allow_cpu: bool = False
    env_id: str = "HalfCheetah-v4"
    act_steps: int = 1


    total_timesteps: int = 1_000_000
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_frequency: int = 2
    utd_ratio: float = 1.0
    alpha_init: float = 0.2


    alpha_min: float = 0.0


    target_entropy_scale: float = 1.0


    target_entropy_scale_final: float = 0.0
    anneal_steps: int = 0
    logp_clip: float = 0.0
    eval_stochastic: int = 0
    grad_clip: float = 0.0


    eval_interval: int = 0
    eval_envs: int = 5
    ckpt_interval: int = 0
    save_full_state: bool = False
    resume_full_state: str = ""
    resume_ckpt: str = ""

    resume_min_buffer: int = 25_000
    u_scale: float = 3.0
    denoising_steps: int = 4
    integration: str = "implicit"
    density_estimator: str = "exact"

    gate_bias: float = 5.0

    lam_jac: float = 0.0
    jac_sigma_target: float = 0.9
    jac_warmup: int = 100_000
    ent_track: bool = False
    ent_margin: float = 0.7
    ent_floor_scale: float = 0.7
    alpha_leak: float = 0.0
    alpha_prior: float = 0.2
    lp_winsor: float = 0.0


    best_reward_threshold_for_success: float = 0.0


    log_interval: int = 1000
    tb_interval: int = 100
    density_diag_every: int = 0
    density_diag_batch: int = 32


def make_env_thunk(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if (capture_video and idx == 0) else None)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk


def make_vec_env(env_id: str, seed: int, num_envs: int, capture_video: bool, run_name: str):
    env_fns = [make_env_thunk(env_id, seed, i, capture_video, run_name) for i in range(num_envs)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    return envs


def native_success_vector(info, num_envs: int):

    for key in ("sacflow_success", "success"):
        if key not in info:
            continue
        value = info[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        flags = np.asarray(value, dtype=bool).reshape(-1)
        if flags.size == 1 and num_envs > 1:
            flags = np.repeat(flags, num_envs)
        if flags.size != num_envs:
            return None
        valid = info.get(f"_{key}")
        if valid is not None:
            valid = np.asarray(valid, dtype=bool).reshape(-1)
            if valid.size == num_envs:
                flags &= valid
        return flags
    return None



def summarize_finished_trajs(
    finished_trajs,
    act_steps: int,
    best_reward_threshold_for_success: float,
):
    if len(finished_trajs) == 0:
        return dict(
            num_episode_finished=0,
            avg_episode_reward=0.0,
            std_episode_reward=0.0,
            avg_best_reward=0.0,
            std_best_reward=0.0,
            success_rate=0.0,
            std_success_rate=0.0,
            avg_episode_length=0.0,
            std_episode_length=0.0,
        )

    episode_reward = np.array([np.sum(tr) for tr in finished_trajs], dtype=np.float32)

    episode_best_reward = np.array([np.max(tr) / act_steps for tr in finished_trajs], dtype=np.float32)
    success_mask = (episode_best_reward >= best_reward_threshold_for_success)

    std_success_mask = (episode_best_reward >= best_reward_threshold_for_success)
    episode_lengths = np.array([len(tr) * act_steps for tr in finished_trajs], dtype=np.float32)

    return dict(
        num_episode_finished=len(finished_trajs),
        avg_episode_reward=float(np.mean(episode_reward)),
        std_episode_reward=float(np.std(episode_reward)),
        avg_best_reward=float(np.mean(episode_best_reward)),
        std_best_reward=float(np.std(episode_best_reward)),
        success_rate=float(np.mean(success_mask)),
        std_success_rate=float(np.std(std_success_mask)),
        avg_episode_length=float(np.mean(episode_lengths)),
        std_episode_length=float(np.std(episode_lengths)),
    )

class replay_buffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: torch.device):
        self.device = device
        self.size = int(size)
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.size, act_dim), dtype=np.float32)
        self.rews = np.zeros((self.size, 1), dtype=np.float32)
        self.dones = np.zeros((self.size, 1), dtype=np.float32)

    def add(self, obs, next_obs, act, rew, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch_size: int):
        max_i = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_i, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        acts = torch.as_tensor(self.acts[idx], device=self.device)
        rews = torch.as_tensor(self.rews[idx], device=self.device)
        dones = torch.as_tensor(self.dones[idx], device=self.device)
        return obs, acts, next_obs, rews, dones


class sinusoidal_pos_emb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        device, dtype = t.device, t.dtype
        scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=dtype) * (-scale))
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def anderson(f, x0, m=5, lam=1e-2, max_iter=25, tol=1e-3, beta=0.7, return_stats=False):
    B, D = x0.shape
    X = torch.zeros(B, m, D, device=x0.device, dtype=x0.dtype)
    Fm = torch.zeros(B, m, D, device=x0.device, dtype=x0.dtype)
    X[:, 0] = x0
    Fm[:, 0] = f(x0)
    X[:, 1] = Fm[:, 0]
    Fm[:, 1] = f(X[:, 1])
    res_curve = []
    z_new = X[:, 1]
    abs_res = (Fm[:, 1] - z_new).norm(dim=1)
    rel_res = abs_res / (Fm[:, 1].norm(dim=1) + 1e-6)
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fm[:, :n] - X[:, :n]
        GTG = torch.bmm(G, G.transpose(1, 2))
        GTG = GTG + lam * torch.eye(n, device=x0.device, dtype=x0.dtype).unsqueeze(0)
        H = torch.zeros(B, n + 1, n + 1, device=x0.device, dtype=x0.dtype)
        H[:, 1:, 1:] = GTG
        H[:, 1:, 1:] += 1e-6 * torch.eye(n, device=x0.device, dtype=x0.dtype).unsqueeze(0)
        H[:, 0, 1:] = 1
        H[:, 1:, 0] = 1
        y = torch.zeros(B, n + 1, 1, device=x0.device, dtype=x0.dtype)
        y[:, 0] = 1
        alpha = torch.linalg.lstsq(H, y).solution[:, 1:, 0]
        z_new = beta * torch.sum(alpha.unsqueeze(-1) * Fm[:, :n], dim=1) + (1.0 - beta) * torch.sum(alpha.unsqueeze(-1) * X[:, :n], dim=1)
        X[:, k % m] = z_new
        Fm[:, k % m] = f(z_new)
        abs_res = (Fm[:, k % m] - z_new).norm(dim=1)
        rel_res = abs_res / (Fm[:, k % m].norm(dim=1) + 1e-6)
        res_mean = rel_res.mean().item()
        res_curve.append(res_mean)
        if res_mean < tol:
            break
    if not return_stats:
        return z_new
    stats = {
        "n_iter": len(res_curve) + 2,
        "res_curve": res_curve,
        "res_last": res_curve[-1] if res_curve else None,
        "res_p50": torch.quantile(rel_res, 0.50).item(),
        "res_p95": torch.quantile(rel_res, 0.95).item(),
        "res_max": rel_res.max().item(),
        "res_fail_frac": (rel_res >= tol).float().mean().item(),
        "abs_res_p95": torch.quantile(abs_res, 0.95).item(),
        "abs_res_max": abs_res.max().item(),
    }
    return z_new, stats


class actor_vector_field_torch(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, time_emb_dim: int = 32, hidden_dim: int = 128,
                 use_layernorm: bool = True, u_scale: float = 3.0, gate_bias: float = 5.0,
                 n_gate_hidden: int = 1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            sinusoidal_pos_emb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        in_dim = obs_dim + action_dim + time_emb_dim
        self.pre = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else None


        def _branch():
            layers = []
            for _ in range(n_gate_hidden):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, action_dim)]
            return nn.Sequential(*layers)
        self.gate = _branch()
        self.cand = _branch()


        self.u_scale = float(u_scale)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))

    def forward(self, a_prev: torch.Tensor, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t = t.to(dtype=obs.dtype)
        temb = self.time_mlp(t)
        h = torch.cat([obs, a_prev, temb], dim=-1)
        h = F.silu(self.pre(h))
        if self.ln is not None:
            h = self.ln(h)
        z = torch.sigmoid(self.gate(h))
        h_tilde = torch.tanh(self.cand(h)) * self.u_scale
        v = z * (h_tilde - a_prev)
        return v


class deq_multistep_flow_actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: torch.Tensor,
        act_high: torch.Tensor,
        *,
        denoising_steps: int = 4,
        time_emb_dim: int = 32,
        hidden_dim: int = 128,
        n_gate_hidden: int = 1,
        use_layernorm: bool = True,
        u_scale: float = 3.0,
        gate_bias: float = 5.0,
        jac_sigma_target: float = 0.0,
        m_fwd: int = 5,
        lam_fwd: float = 1e-2,
        max_iter_fwd: int = 15,
        tol_fwd: float = 1e-3,
        beta_fwd: float = 0.7,
        m_bwd: int = 5,
        lam_bwd: float = 1e-2,
        max_iter_bwd: int = 10,
        tol_bwd: float = 1e-3,
        beta_bwd: float = 0.7,
        z_std: float = 1.0,
        logdet_eps: float = 1e-6,
        exact_backward: bool = False,
        integration: str = "implicit",
        density_estimator: str = "exact",
        density_diag_every: int = 0,
        density_diag_batch: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.T = denoising_steps
        self.dt = 1.0 / float(self.T)
        self.vf = actor_vector_field_torch(
            obs_dim, act_dim, time_emb_dim=time_emb_dim, hidden_dim=hidden_dim,
            use_layernorm=use_layernorm, u_scale=u_scale, gate_bias=gate_bias,
            n_gate_hidden=n_gate_hidden,
        )
        self.jac_sigma_target = float(jac_sigma_target)
        self._jac_pen = None
        lower = torch.tril(torch.ones(self.T, self.T, dtype=torch.float32))
        Ik = torch.eye(self.act_dim, dtype=torch.float32)
        M = torch.kron(lower, Ik)
        self.register_buffer("matrix_in_g", M)
        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("action_scale", (act_high - act_low) / 2.0)
        self.register_buffer("action_bias", (act_high + act_low) / 2.0)
        self.m_fwd, self.lam_fwd, self.max_iter_fwd, self.tol_fwd, self.beta_fwd = m_fwd, lam_fwd, max_iter_fwd, tol_fwd, beta_fwd
        self.m_bwd, self.lam_bwd, self.max_iter_bwd, self.tol_bwd, self.beta_bwd = m_bwd, lam_bwd, max_iter_bwd, tol_bwd, beta_bwd
        self.z_std = float(z_std)
        self.logdet_eps = float(logdet_eps)
        self.exact_backward = bool(exact_backward)
        assert integration in ("implicit", "explicit"), integration
        self.integration = integration
        assert density_estimator == "exact", density_estimator
        self.density_estimator = density_estimator
        self.last_fwd = {}
        self.last_bwd = {}
        self.last_diag = {}
        self._diag_tick = 0
        self.density_diag_every = int(density_diag_every)
        self.density_diag_batch = int(density_diag_batch)
        self._matrix_diag_cache = {}

    def _sample_base_z(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(B, self.act_dim, device=device, dtype=dtype) * self.z_std

    def _make_a0_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return z.repeat(1, self.T)

    def _rollout_explicit(self, obs: torch.Tensor, z: torch.Tensor):


        u = z
        pre = []
        for k in range(self.T):
            t = torch.full((u.shape[0], 1), k * self.dt, device=u.device, dtype=u.dtype)
            pre.append(u)
            u = u + self.dt * self.vf(u, obs, t)
        return torch.stack(pre, dim=1), u

    def _g(self, z_flat: torch.Tensor, a0: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        B = z_flat.shape[0]
        T = self.T
        k = self.act_dim
        device, dtype = z_flat.device, z_flat.dtype
        z_all = z_flat.view(B, T, k)
        a_flat = z_all.reshape(B * T, k)
        obs_rep = obs.unsqueeze(1).expand(B, T, obs.shape[-1]).reshape(B * T, obs.shape[-1])
        t_vals = (torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(-1) * self.dt)
        t_rep = t_vals.unsqueeze(0).expand(B, T, 1).reshape(B * T, 1)
        v_flat = self.vf(a_flat, obs_rep, t_rep)
        v_all = v_flat.view(B, T, k).reshape(B, T * k)
        M = self.matrix_in_g.to(device=device, dtype=dtype)
        return a0 + (self.dt * v_all) @ M.t()

    def _solve_zstar_with_implicit_grad(self, obs: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:


        z0 = a0.detach().clone()

        def g_closure(z):
            return self._g(z, a0, obs)

        with torch.no_grad():
            z_star, fwd_stats = anderson(
                g_closure,
                z0,
                m=self.m_fwd,
                lam=self.lam_fwd,
                max_iter=self.max_iter_fwd,
                tol=self.tol_fwd,
                beta=self.beta_fwd,
                return_stats=True,
            )
            self.last_fwd = fwd_stats

        if torch.is_grad_enabled():
            z_solved = z_star
            z_star = g_closure(z_solved)

            with torch.no_grad():
                attach_delta = z_star.detach() - z_solved
                attach_rel = attach_delta.norm(dim=1) / (z_solved.norm(dim=1) + 1e-6)
                solved_end = z_solved.view(z_solved.shape[0], self.T, self.act_dim)[:, -1]
                attached_end = z_star.detach().view(z_star.shape[0], self.T, self.act_dim)[:, -1]
                attach_action_l2 = (
                    (torch.tanh(attached_end) - torch.tanh(solved_end)) * self.action_scale
                ).norm(dim=1)
                self.last_fwd.update({
                    "attach_rel_mean": attach_rel.mean().item(),
                    "attach_rel_p95": torch.quantile(attach_rel, 0.95).item(),
                    "attach_rel_max": attach_rel.max().item(),
                    "attach_rel_gt_tol_frac": (attach_rel >= self.tol_fwd).float().mean().item(),
                    "attach_action_l2_p95": torch.quantile(attach_action_l2, 0.95).item(),
                    "attach_action_l2_max": attach_action_l2.max().item(),
                })


            if self.exact_backward:


                B, D = z_star.shape
                T, k = self.T, self.act_dim
                device, dtype = z_star.device, z_star.dtype
                with torch.no_grad():
                    u_all = z_star.detach().view(B, T, k).reshape(B * T, k)
                    obs_all = obs.unsqueeze(1).expand(B, T, obs.shape[-1]).reshape(B * T, obs.shape[-1])
                    t_vals = torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(-1) * self.dt
                    t_all = t_vals.unsqueeze(0).expand(B, T, 1).reshape(B * T, 1)

                    def vf_single(u_s, o_s, t_s):
                        return self.vf(u_s.unsqueeze(0), o_s.unsqueeze(0), t_s.unsqueeze(0)).squeeze(0)

                    J = vmap(jacrev(vf_single, argnums=0))(u_all, obs_all, t_all).view(B, T, k, k)
                    BD = torch.zeros(B, D, D, device=device, dtype=dtype)
                    for t in range(T):
                        BD[:, t * k:(t + 1) * k, t * k:(t + 1) * k] = J[:, t]
                    M = self.matrix_in_g.to(device=device, dtype=dtype)
                    I_D = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
                    A_T = (I_D - self.dt * (M.unsqueeze(0) @ BD)).transpose(1, 2) + 1e-6 * I_D

                def backward_hook(grad_output):
                    self.last_bwd = {"n_iter": 0, "res_last": 0.0}
                    return torch.linalg.solve(A_T, grad_output.unsqueeze(-1)).squeeze(-1)

                if z_star.requires_grad:
                    z_star.register_hook(backward_hook)
                return z_star

            z0g = z_star.detach().clone().requires_grad_(True)
            g0 = g_closure(z0g)

            def backward_hook(grad_output):
                def lin_map(u):
                    vjp = torch.autograd.grad(g0, z0g, u, retain_graph=True)[0]
                    return grad_output + vjp

                u_star, bwd_stats = anderson(
                    lin_map,
                    grad_output,
                    m=self.m_bwd,
                    lam=self.lam_bwd,
                    max_iter=self.max_iter_bwd,
                    tol=self.tol_bwd,
                    beta=self.beta_bwd,
                    return_stats=True,
                )
                self.last_bwd = bwd_stats
                return u_star

            if z_star.requires_grad:
                z_star.register_hook(backward_hook)

        return z_star

    def _forward_core(
        self,
        obs: torch.Tensor,
        need_logprob: bool,
        need_jacobian_penalty: bool = False,
    ):
        B = obs.shape[0]
        device, dtype = obs.device, obs.dtype
        T = self.T
        act_dim = self.act_dim

        z = self._sample_base_z(B, device, dtype)

        if self.integration == "explicit":
            u_pre, u_T = self._rollout_explicit(obs, z)
        else:
            a0 = self._make_a0_from_z(z).to(device=device, dtype=dtype)
            z_star = self._solve_zstar_with_implicit_grad(obs, a0)
            a_seq = z_star.view(B, T, act_dim)
            u_T = a_seq[:, -1, :]

        a_T = torch.tanh(u_T)
        a_env = a_T * self.action_scale + self.action_bias

        if not need_logprob and not need_jacobian_penalty:
            return a_env, None

        if need_logprob:

            logp_z = -0.5 * (z.pow(2).sum(dim=-1) / (self.z_std ** 2) + act_dim * math.log(2.0 * math.pi * (self.z_std ** 2)))


        if self.integration == "explicit":
            u_all = u_pre.flatten(0, 1)
            t_vals = (torch.arange(0, T, device=device, dtype=dtype).unsqueeze(-1) * self.dt)
        else:
            u_all = a_seq.flatten(0, 1)
            t_vals = (torch.arange(1, T + 1, device=device, dtype=dtype).unsqueeze(-1) * self.dt)
        obs_all = obs.unsqueeze(1).expand(B, T, obs.shape[-1]).reshape(B * T, obs.shape[-1])
        t_all = t_vals.unsqueeze(0).expand(B, T, 1).reshape(B * T, 1)

        def vf_single(u_single, o_single, t_single):
            out = self.vf(
                u_single.unsqueeze(0),
                o_single.unsqueeze(0),
                t_single.unsqueeze(0),
            )
            return out.squeeze(0)

        J = vmap(jacrev(vf_single, argnums=0))(u_all, obs_all, t_all)


        if self.jac_sigma_target > 0 and torch.is_grad_enabled():
            dtJ_spec_g = torch.linalg.matrix_norm(self.dt * J, ord=2)
            self._jac_pen = F.relu(dtJ_spec_g - self.jac_sigma_target).pow(2).view(B, T).sum(dim=1)


        matrix_probe = None
        with torch.no_grad():
            dtJ_det = (self.dt * J).detach()
            dtJ_spec = torch.linalg.matrix_norm(dtJ_det, ord=2)
            sech2_raw = 1.0 - a_T.detach().pow(2)
            self._diag_tick += 1
            self.last_diag = {
                "dtJ_specnorm_mean": dtJ_spec.mean().item(),
                "dtJ_specnorm_p50": torch.quantile(dtJ_spec, 0.50).item(),
                "dtJ_specnorm_p95": torch.quantile(dtJ_spec, 0.95).item(),
                "dtJ_specnorm_max": dtJ_spec.max().item(),
                "dtJ_noncontractive_frac": (dtJ_spec >= 1.0).float().mean().item(),
                "sech2_raw_min": sech2_raw.min().item(),
                "sech2_raw_p01": torch.quantile(sech2_raw, 0.01).item(),
                "tanh_guard_frac": (sech2_raw <= 1e-6).float().mean().item(),
                "tanh_saturation_0999_frac": (a_T.detach().abs() >= 0.999).float().mean().item(),
            }

            diag_due = (
                self.density_diag_every > 0
                and self._diag_tick % self.density_diag_every == 0
            )
            if diag_due:
                n_probe = min(self.density_diag_batch, B)
                J_probe = J.detach().view(B, T, act_dim, act_dim)[:n_probe].reshape(-1, act_dim, act_dim)
                M_raw = self.dt * J_probe if self.integration == "explicit" else -self.dt * J_probe
                M_raw = M_raw.clone()
                idx_probe = torch.arange(act_dim, device=device)
                M_raw[:, idx_probe, idx_probe] += 1.0
                M_guard = M_raw.clone()
                M_guard[:, idx_probe, idx_probe] += self.logdet_eps
                sign_raw, ld_raw = torch.linalg.slogdet(M_raw)
                sign_guard, ld_guard = torch.linalg.slogdet(M_guard)
                singular_values = torch.linalg.svdvals(M_raw)
                sigma_min = singular_values[:, -1]
                sigma_max = singular_values[:, 0]
                condition = sigma_max / sigma_min.clamp_min(torch.finfo(sigma_min.dtype).tiny)
                ld_delta = (ld_guard - ld_raw).abs()
                eigvals = torch.linalg.eigvals(self.dt * J_probe.cpu())
                self._matrix_diag_cache = {
                    "matrix_probe_tick": float(self._diag_tick),
                    "unguarded_sigma_min_mean": sigma_min.mean().item(),
                    "unguarded_sigma_min_p05": torch.quantile(sigma_min, 0.05).item(),
                    "unguarded_sigma_min_min": sigma_min.min().item(),
                    "unguarded_condition_p50": torch.quantile(condition, 0.50).item(),
                    "unguarded_condition_p95": torch.quantile(condition, 0.95).item(),
                    "unguarded_condition_max": condition.max().item(),
                    "unguarded_near_singular_1e3_frac": (sigma_min <= 1e-3).float().mean().item(),
                    "unguarded_near_singular_1e6_frac": (sigma_min <= 1e-6).float().mean().item(),
                    "unguarded_negdet_frac": (sign_raw < 0).float().mean().item(),
                    "unguarded_slogdet_zero_frac": (sign_raw == 0).float().mean().item(),
                    "logdet_eps_delta_abs_mean": ld_delta.mean().item(),
                    "logdet_eps_delta_abs_p95": torch.quantile(ld_delta, 0.95).item(),
                    "logdet_eps_delta_abs_max": ld_delta.max().item(),
                    "logdet_eps_sign_flip_frac": (sign_raw != sign_guard).float().mean().item(),
                    "dtJ_eig_re_min": eigvals.real.min().item(),
                    "dtJ_eig_re_max": eigvals.real.max().item(),
                    "imp_dist_to_sing_min": (1.0 - eigvals).abs().min().item(),
                }
                matrix_probe = {
                    "n": n_probe,
                    "sign_raw": sign_raw,
                    "ld_raw": ld_raw,
                }

            self.last_diag.update(self._matrix_diag_cache)

        if not need_logprob:
            return a_env, None

        idx = torch.arange(act_dim, device=device)
        if self.integration == "explicit":
            M = self.dt * J
        else:
            M = -self.dt * J
        M[:, idx, idx] += (1.0 + self.logdet_eps)
        sign, logabsdet_raw = torch.linalg.slogdet(M)
        logabsdet = torch.clamp(logabsdet_raw, min=-50.0, max=50.0)
        logabsdet = logabsdet.view(B, T).sum(dim=1)
        log_volume_term = -logabsdet if self.integration == "explicit" else logabsdet


        with torch.no_grad():
            ld = logabsdet_raw.detach()
            self.last_diag.update({
                "logdet_step_mean": ld.mean().item(),
                "logdet_step_min": ld.min().item(),
                "logdet_step_max": ld.max().item(),
                "logdet_clamp_frac": (ld.abs() > 50.0).float().mean().item(),
                "negdet_frac": (sign <= 0).float().mean().item(),
            })


            if self._diag_tick % 50 == 0:
                if self.density_diag_every <= 0:
                    ev = torch.linalg.eigvals((self.dt * J).detach().cpu())
                    self.last_diag.update({
                        "dtJ_eig_re_min": ev.real.min().item(),
                        "dtJ_eig_re_max": ev.real.max().item(),
                        "imp_dist_to_sing_min": (1.0 - ev).abs().min().item(),
                    })


        scale = self.action_scale
        eps = 1e-6
        log_da_du = torch.log(torch.clamp(1.0 - a_T.pow(2), min=eps))
        log_scale = torch.log(torch.abs(scale) + eps)
        log_scale = log_scale.expand_as(log_da_du) if log_scale.ndim == 0 else log_scale.view(1, -1).expand_as(log_da_du)
        tanh_scale_logdet = (log_scale + log_da_du).sum(dim=-1)

        logp = logp_z + log_volume_term - tanh_scale_logdet

        with torch.no_grad():
            logp_det = logp.detach()
            self.last_diag.update({
                "logp_raw_mean": logp_det.mean().item(),
                "logp_raw_p05": torch.quantile(logp_det, 0.05).item(),
                "logp_raw_p50": torch.quantile(logp_det, 0.50).item(),
                "logp_raw_p95": torch.quantile(logp_det, 0.95).item(),
                "logp_raw_min": logp_det.min().item(),
                "logp_raw_max": logp_det.max().item(),
                "logp_raw_nonfinite_frac": (~torch.isfinite(logp_det)).float().mean().item(),
            })
            if matrix_probe is not None:
                n_probe = matrix_probe["n"]
                raw_step = matrix_probe["ld_raw"].view(n_probe, T).sum(dim=1)
                raw_volume = -raw_step if self.integration == "explicit" else raw_step
                raw_sech = 1.0 - a_T[:n_probe].detach().pow(2)
                raw_squash = (
                    torch.log(torch.abs(scale))
                    + torch.log(raw_sech)
                ).sum(dim=-1)
                unguarded_logp = logp_z[:n_probe].detach() + raw_volume - raw_squash
                finite = torch.isfinite(unguarded_logp)
                guard_delta = (logp_det[:n_probe] - unguarded_logp).abs()
                finite_delta = guard_delta[finite]
                self._matrix_diag_cache.update({
                    "unguarded_logp_nonfinite_frac": (~finite).float().mean().item(),
                    "unguarded_logp_finite_frac": finite.float().mean().item(),
                    "guarded_vs_unguarded_logp_abs_mean": (
                        finite_delta.mean().item() if finite_delta.numel() else 0.0
                    ),
                    "guarded_vs_unguarded_logp_abs_p95": (
                        torch.quantile(finite_delta, 0.95).item() if finite_delta.numel() else 0.0
                    ),
                    "guarded_vs_unguarded_logp_abs_max": (
                        finite_delta.max().item() if finite_delta.numel() else 0.0
                    ),
                })
                self.last_diag.update(self._matrix_diag_cache)

        return a_env, logp

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a_env, _ = self._forward_core(obs, need_logprob=False)
        return a_env

    def forward_with_logprob(self, obs: torch.Tensor):
        return self._forward_core(obs, need_logprob=True)

    def forward_with_jacobian_penalty(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self._forward_core(
            obs,
            need_logprob=False,
            need_jacobian_penalty=True,
        )
        return action

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if not deterministic:
            return self.forward(obs)
        B = obs.shape[0]
        device, dtype = obs.device, obs.dtype
        z = torch.zeros(B, self.act_dim, device=device, dtype=dtype)
        if self.integration == "explicit":
            _, u = self._rollout_explicit(obs, z)
            return torch.tanh(u) * self.action_scale + self.action_bias
        a0 = self._make_a0_from_z(z)

        def g_closure(z_flat):
            return self._g(z_flat, a0, obs)

        z_star = anderson(
            g_closure,
            a0.clone(),
            m=self.m_fwd,
            lam=self.lam_fwd,
            max_iter=self.max_iter_fwd,
            tol=self.tol_fwd,
            beta=self.beta_fwd,
            return_stats=False,
        )
        a_seq = z_star.view(B, self.T, self.act_dim)
        u = a_seq[:, -1, :]
        a = torch.tanh(u)
        a_env = a * self.action_scale + self.action_bias
        return a_env

class critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.data.mul_(1.0 - tau).add_(p_s.data, alpha=tau)

def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return float(x.item())

def format_kv(kvs: dict) -> str:
    keys = ["step", "sps", "ep_ret", "ep_len", "q1", "q2", "q_loss", "pi_loss", "buffer", "alpha", "logp", "qpi"]
    parts = []
    for k in keys:
        if k in kvs and kvs[k] is not None and not (isinstance(kvs[k], float) and (math.isnan(kvs[k]) or math.isinf(kvs[k]))):
            v = kvs[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")
    for k, v in kvs.items():
        if k not in keys and v is not None:
            parts.append(f"{k}={v}")
    return " | " + " ".join(parts)


def select_device(args, run_name: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; pass allow_cpu=true to run on CPU.")
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
    cuda_version = torch.version.cuda or "none"
    print(
        "launch: | "
        f"run_name={run_name} env_id={args.env_id} exp_name={args.exp_name} seed={args.seed} "
        f"density_estimator={args.density_estimator} "
        f"device={device} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"gpu={gpu_name} cuda={cuda_version} torch={torch.__version__} allow_cpu={args.allow_cpu}"
    )
    return device

@torch.no_grad()
def eval_policy(eval_env, actor_net, device, num_envs: int, eval_seed=None,
                stochastic: bool = False) -> float:

    obs, _ = eval_env.reset(seed=eval_seed)
    obs = obs.astype(np.float32).reshape(num_envs, -1)
    returns = np.zeros(num_envs)
    active = np.ones(num_envs, dtype=bool)
    successes = np.zeros(num_envs, dtype=bool)
    native_success_seen = False
    while active.any():
        if hasattr(actor_net, "act"):
            a = actor_net.act(torch.as_tensor(obs, device=device), deterministic=not stochastic)
        else:
            a = actor_net(torch.as_tensor(obs, device=device))
        obs, rew, term, trunc, info = eval_env.step(a.cpu().numpy().astype(np.float32))
        obs = obs.astype(np.float32).reshape(num_envs, -1)
        returns += np.asarray(rew) * active
        native_success = native_success_vector(info, num_envs)
        if native_success is not None:
            successes |= native_success & active
            native_success_seen = True
        active &= ~(np.asarray(term) | np.asarray(trunc))
    mean_return = float(returns.mean())
    eval_env._sacflow_last_native_success_rate = (
        float(successes.mean()) if native_success_seen else None
    )
    return mean_return


def _atomic_torch_save(state, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _save_step_checkpoint(state, run_dir: str, step: int):

    step_path = os.path.join(run_dir, f"ckpt_step_{step:09d}.pt")
    latest_path = os.path.join(run_dir, "ckpt.pt")
    latest_tmp = f"{latest_path}.tmp"
    _atomic_torch_save(state, step_path)
    try:
        if os.path.lexists(latest_tmp):
            os.unlink(latest_tmp)
        os.link(step_path, latest_tmp)
        os.replace(latest_tmp, latest_path)
    finally:
        if os.path.lexists(latest_tmp):
            os.unlink(latest_tmp)


ACTOR_REGISTRY = {}


def _actor_class_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_state(v) for v in value)
    return copy.deepcopy(value)


def _rng_state():
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(), "python": random.getstate()}


def _restore_rng(state):
    _require_keys(state, ("torch_cpu", "torch_cuda", "numpy", "python"), "rng")
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _same_rng(a, b):
    return (torch.equal(a["torch_cpu"], b["torch_cpu"])
            and len(a["torch_cuda"]) == len(b["torch_cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"]))
            and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]
            and a["python"] == b["python"])


def _require_keys(state, keys, label):
    if not isinstance(state, dict) or set(keys) - state.keys():
        raise ValueError(f"{label}: missing required keys {sorted(keys)}")






def _validate_full_state(state, args, obs_dim, act_dim, auto_alpha):
    _require_keys(state, ("actor", "q1", "q2", "log_alpha", "global_step", "args",
                         "replay", "opt_states", "rng", "full_state", "actor_class",
                         "q1_targ", "q2_targ", "loop", "collector", "actor_runtime"),
                  "full-state checkpoint")
    if state["full_state"] is not True:
        raise ValueError("checkpoint is not a full-state checkpoint")
    _require_keys(state["args"], ("env_id", "seed", "num_envs"), "checkpoint args")
    for key in ("env_id", "seed", "num_envs"):
        if state["args"][key] != getattr(args, key):
            raise ValueError(f"full-state {key} mismatch")


    runtime_args = {"exp_name", "run_name", "total_timesteps", "save_model", "save_full_state",
                    "resume_full_state", "resume_ckpt",
                    "resume_min_buffer", "allow_cpu"}
    for key, value in vars(args).items():
        if key not in runtime_args and state["args"].get(key) != value:
            raise ValueError(f"full-state {key} mismatch: retain the saved training recipe")
    replay = state["replay"]
    _require_keys(replay, ("obs", "acts", "rews", "next_obs", "done", "ptr", "size", "full"), "replay")
    for key, dim in (("obs", obs_dim), ("next_obs", obs_dim), ("acts", act_dim),
                     ("rews", 1), ("done", 1)):
        if np.shape(replay[key]) != (args.buffer_size, dim):
            raise ValueError(f"replay {key}: observation/action dimension or capacity mismatch")
    if replay["size"] != args.buffer_size or not 0 <= replay["ptr"] < replay["size"]:
        raise ValueError("invalid replay size/pointer")
    _require_keys(state["opt_states"], ("actor_opt", "q_opt") + (("alpha_opt",) if auto_alpha else ()), "optimizers")
    for name, opt in state["opt_states"].items():
        _require_keys(opt, ("state", "param_groups"), name)
    _require_keys(state["rng"], ("torch_cpu", "torch_cuda", "numpy", "python"), "rng")
    _require_keys(state["loop"], ("n_updates", "last_eval", "last_log", "last_tb", "last_ckpt",
                                   "ent_ema", "target_entropy", "cur_rew_trajs",
                                   "finished_trajs_window"), "loop")
    collector = state["collector"]
    _require_keys(collector, ("reset_rng", "actions", "step_rng", "action_rng", "obs", "prev_autoreset"), "collector")
    if (np.shape(collector["obs"]) != (args.num_envs, obs_dim)
            or np.shape(collector["actions"]) != (state["global_step"] // args.num_envs, args.num_envs, act_dim)
            or np.shape(collector["prev_autoreset"]) != (args.num_envs,)):
        raise ValueError("collector observation/action dimension or step mismatch")


def main(args: Args):
    if args.save_full_state and args.resume_ckpt and not args.resume_full_state:
        raise ValueError("full-state saving cannot reconstruct history from a seam checkpoint")
    run_name = args.run_name or (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4().hex[:12]}"
    )
    assert args.density_estimator == "exact", args.density_estimator
    auto_alpha = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = select_device(args, run_name)
    writer = SummaryWriter(f"runs/{run_name}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    env = make_vec_env(args.env_id, args.seed, args.num_envs, args.capture_video, run_name)

    single_obs_space = env.single_observation_space
    single_act_space = env.single_action_space
    assert isinstance(single_act_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.prod(single_obs_space.shape))
    act_dim = int(np.prod(single_act_space.shape))
    act_low = single_act_space.low.astype(np.float32)
    act_high = single_act_space.high.astype(np.float32)

    full_mode = args.save_full_state or bool(args.resume_full_state)
    full_ckpt = None
    actor_cls = deq_multistep_flow_actor
    if full_mode:
        ACTOR_REGISTRY[_actor_class_name(actor_cls)] = actor_cls
    if args.resume_full_state:
        full_ckpt = torch.load(args.resume_full_state, map_location="cpu", weights_only=False)
        _validate_full_state(full_ckpt, args, obs_dim, act_dim, auto_alpha)
        if full_ckpt["actor_class"] not in ACTOR_REGISTRY:
            raise ValueError(f"unregistered checkpoint actor class: {full_ckpt['actor_class']}")
        actor_cls = ACTOR_REGISTRY[full_ckpt["actor_class"]]
    actor_net = actor_cls(
        obs_dim, act_dim, act_low, act_high,
        denoising_steps=args.denoising_steps, integration=args.integration,
        density_estimator=args.density_estimator,
        density_diag_every=args.density_diag_every,
        density_diag_batch=args.density_diag_batch,
        gate_bias=args.gate_bias,
        jac_sigma_target=(args.jac_sigma_target if args.lam_jac > 0 else 0.0),
        hidden_dim=128, u_scale=args.u_scale,
        max_iter_fwd=15, tol_fwd=1e-3, lam_fwd=1e-2, beta_fwd=0.7,
        max_iter_bwd=10, tol_bwd=1e-3, lam_bwd=1e-2, beta_bwd=0.7,
    ).to(device)

    q1 = critic(obs_dim, act_dim).to(device)
    q2 = critic(obs_dim, act_dim).to(device)
    q1_targ = critic(obs_dim, act_dim).to(device)
    q2_targ = critic(obs_dim, act_dim).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.critic_lr)


    actor_rail_lower = target_rail_lower = -args.logp_clip if args.logp_clip > 0 else -math.inf
    actor_rail_upper = target_rail_upper = args.logp_clip if args.logp_clip > 0 else math.inf
    alpha_init = args.alpha_init

    target_entropy = -float(act_dim) * args.target_entropy_scale
    log_alpha = torch.tensor(
        math.log(alpha_init),
        device=device,
        requires_grad=auto_alpha,
    )
    _resume_step = 0
    if args.resume_ckpt and not args.resume_full_state:
        _ck = torch.load(args.resume_ckpt, map_location=device)
        actor_net.load_state_dict(_ck["actor"])
        q1.load_state_dict(_ck["q1"])
        q2.load_state_dict(_ck["q2"])
        q1_targ.load_state_dict(q1.state_dict())
        q2_targ.load_state_dict(q2.state_dict())
        with torch.no_grad():
            log_alpha.data.copy_(_ck["log_alpha"].to(device).reshape(log_alpha.shape))
        _resume_step = int(_ck["global_step"])
        print(f"seam-resume: loaded {args.resume_ckpt} @ step {_resume_step}")
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr) if auto_alpha else None
    alpha = log_alpha.exp().item()

    rb = replay_buffer(obs_dim, act_dim, args.buffer_size, device)

    eval_env = None
    last_eval = 0
    if args.eval_interval > 0:
        eval_env = make_vec_env(args.env_id, args.seed + 10_000, args.eval_envs, False, run_name)

    if full_mode:
        reset_rng = full_ckpt["collector"]["reset_rng"] if full_ckpt else _rng_state()
        _restore_rng(reset_rng)
        collector_actions, collector_rng = [], {}
    obs, info = env.reset(seed=args.seed)
    obs = obs.astype(np.float32).reshape(args.num_envs, -1)

    cur_rew_trajs = [[] for _ in range(args.num_envs)]
    finished_trajs_window = []


    prev_autoreset = np.zeros(args.num_envs, dtype=bool)

    global_step = _resume_step
    n_updates = 0
    last_log = 0
    last_tb = 0
    last_ckpt = 0
    last_q1 = last_q2 = last_q_loss = last_pi_loss = last_qpi = last_logp = last_gnorm = None
    last_logp_raw_mean = last_logp_raw_p05 = last_logp_raw_p95 = None
    last_logp_clip_frac = last_logp_clip_excess = None
    last_jac_pen = None
    ent_ema = None
    start_time = time.perf_counter()

    if full_ckpt is not None:
        c = full_ckpt["collector"]


        collector_actions = list(c["actions"])
        collector_rng = c["step_rng"]
        for index, action in enumerate(collector_actions):
            if index in collector_rng:
                _restore_rng(collector_rng[index])
            obs, _, _, _, _ = env.step(action)
        obs = obs.astype(np.float32).reshape(args.num_envs, -1)
        if not np.array_equal(obs, c["obs"]):
            raise ValueError("environment action replay did not reproduce checkpoint observations exactly")
        env.action_space.np_random.bit_generator.state = c["action_rng"]
        prev_autoreset = c["prev_autoreset"].copy()
        for name, net in (("actor", actor_net), ("q1", q1), ("q2", q2),
                          ("q1_targ", q1_targ), ("q2_targ", q2_targ)):
            try:
                net.load_state_dict(full_ckpt[name])
            except (RuntimeError, KeyError) as exc:
                raise ValueError(f"invalid full-state {name}: {exc}") from exc
        for name, value in full_ckpt["actor_runtime"].items():
            setattr(actor_net, name, value)
        with torch.no_grad():
            log_alpha.copy_(full_ckpt["log_alpha"].to(device))
        for name, opt in (("actor_opt", actor_opt), ("q_opt", q_opt), ("alpha_opt", alpha_opt)):
            if opt is not None:
                opt.load_state_dict(full_ckpt["opt_states"][name])
        for name in ("obs", "acts", "rews", "next_obs", "done"):
            getattr(rb, "dones" if name == "done" else name)[:] = full_ckpt["replay"][name]
        rb.ptr, rb.full = full_ckpt["replay"]["ptr"], full_ckpt["replay"]["full"]
        global_step = int(full_ckpt["global_step"])
        loop = full_ckpt["loop"]
        n_updates, last_eval, last_log, last_tb, last_ckpt = (
            loop[k] for k in ("n_updates", "last_eval", "last_log", "last_tb", "last_ckpt"))
        ent_ema, target_entropy = (loop[k] for k in ("ent_ema", "target_entropy"))
        cur_rew_trajs, finished_trajs_window = loop["cur_rew_trajs"], loop["finished_trajs_window"]
        alpha = log_alpha.exp().item()
        _restore_rng(full_ckpt["rng"])
        print(f"full-state resume: loaded {args.resume_full_state} @ step {global_step}")

    def full_state_checkpoint():
        return _cpu_state({
            "actor": actor_net.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
            "log_alpha": log_alpha, "global_step": global_step, "args": vars(args),
            "full_state": True, "actor_class": _actor_class_name(type(actor_net)),
            "q1_targ": q1_targ.state_dict(), "q2_targ": q2_targ.state_dict(),
            "replay": {"obs": rb.obs, "acts": rb.acts, "rews": rb.rews,
                       "next_obs": rb.next_obs, "done": rb.dones,
                       "ptr": rb.ptr, "size": rb.size, "full": rb.full},
            "opt_states": {name: opt.state_dict() for name, opt in
                           (("actor_opt", actor_opt), ("q_opt", q_opt), ("alpha_opt", alpha_opt))
                           if opt is not None},
            "rng": _rng_state(),
            "actor_runtime": {name: getattr(actor_net, name) for name in
                              ("_diag_tick",) if hasattr(actor_net, name)},
            "loop": dict(n_updates=n_updates, last_eval=last_eval, last_log=last_log,
                         last_tb=last_tb, last_ckpt=last_ckpt, ent_ema=ent_ema,
                         target_entropy=target_entropy,
                         cur_rew_trajs=cur_rew_trajs, finished_trajs_window=finished_trajs_window),
            "collector": dict(reset_rng=reset_rng, actions=np.asarray(collector_actions, dtype=np.float32).reshape(-1, args.num_envs, act_dim),
                              step_rng=collector_rng, action_rng=env.action_space.np_random.bit_generator.state,
                              obs=obs, prev_autoreset=prev_autoreset),
        })

    while global_step < args.total_timesteps:
        if global_step < args.learning_starts:
            act = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device)
                act_t = actor_net(obs_t).cpu().numpy()
            act = np.clip(act_t, act_low, act_high).astype(np.float32)

        if full_mode:
            step_rng = _rng_state()
        next_obs, rew, term, trunc, info = env.step(act)
        if full_mode:

            if not _same_rng(step_rng, _rng_state()):
                collector_rng[len(collector_actions)] = step_rng
            collector_actions.append(act.copy())
        next_obs = next_obs.astype(np.float32).reshape(args.num_envs, -1)
        term = np.asarray(term)
        trunc = np.asarray(trunc)

        for i in range(args.num_envs):
            if prev_autoreset[i]:
                continue
            rb.add(
                obs[i],
                next_obs[i],
                act[i],
                np.array([rew[i]], dtype=np.float32),
                np.array([float(term[i])], dtype=np.float32),
            )
            cur_rew_trajs[i].append(float(rew[i]))
            if bool(term[i]) or bool(trunc[i]):
                finished_trajs_window.append(cur_rew_trajs[i])
                cur_rew_trajs[i] = []

        prev_autoreset = np.logical_or(term, trunc)
        obs = next_obs
        global_step += args.num_envs

        if global_step >= args.learning_starts and len(rb) >= max(args.batch_size, args.resume_min_buffer if args.resume_ckpt and not args.resume_full_state else 0):
            num_updates = max(1, int(round(args.num_envs * args.utd_ratio)))
            for _ in range(num_updates):
                b_obs, b_act, b_next_obs, b_rew, b_done = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_act, next_logp = actor_net.forward_with_logprob(b_next_obs)
                    if target_rail_lower > -math.inf or target_rail_upper < math.inf:
                        next_logp = next_logp.clamp(target_rail_lower, target_rail_upper)
                    alpha_t = log_alpha.exp()

                with torch.no_grad():
                    q1_next = q1_targ(b_next_obs, next_act)
                    q2_next = q2_targ(b_next_obs, next_act)
                    q_min_next = torch.min(q1_next, q2_next)

                    q_next = q_min_next - alpha_t * next_logp.unsqueeze(-1)
                    target_q = b_rew + (1.0 - b_done) * args.gamma * q_next

                q1_pred = q1(b_obs, b_act)
                q2_pred = q2(b_obs, b_act)
                if not bool(torch.isfinite(target_q).all()):
                    raise FloatingPointError(f"step {global_step}: non-finite TD target")
                q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
                if not bool(torch.isfinite(q_loss)):
                    raise FloatingPointError(f"step {global_step}: non-finite q_loss")

                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                q_opt.step()

                n_updates += 1
                last_q1 = _to_float(q1_pred.mean())
                last_q2 = _to_float(q2_pred.mean())
                last_q_loss = _to_float(q_loss)

                if n_updates % args.policy_frequency == 0:
                    if args.target_entropy_scale_final > 0:
                        frac = min(1.0, global_step / (args.anneal_steps or args.total_timesteps))
                        scale = args.target_entropy_scale + (args.target_entropy_scale_final - args.target_entropy_scale) * frac
                        target_entropy = -float(act_dim) * scale
                    act_pi, act_logp = actor_net.forward_with_logprob(b_obs)
                    with torch.no_grad():
                        raw_lp = act_logp.detach()
                        last_logp_raw_mean = _to_float(raw_lp.mean())
                        last_logp_raw_p05 = _to_float(torch.quantile(raw_lp, 0.05))
                        last_logp_raw_p95 = _to_float(torch.quantile(raw_lp, 0.95))
                        if args.logp_clip > 0:
                            last_logp_clip_frac = _to_float((raw_lp.abs() > args.logp_clip).float().mean())
                            last_logp_clip_excess = _to_float(
                                F.relu(raw_lp.abs() - args.logp_clip).mean()
                            )
                    if actor_rail_lower > -math.inf or actor_rail_upper < math.inf:
                        act_logp = act_logp.clamp(actor_rail_lower, actor_rail_upper)

                    q_pi = torch.min(q1(b_obs, act_pi), q2(b_obs, act_pi))

                    alpha_t = log_alpha.exp().detach()
                    pi_loss = (alpha_t * act_logp - q_pi.squeeze(-1)).mean()
                    if args.lam_jac > 0 and getattr(actor_net, "_jac_pen", None) is not None:
                        lam_jac_t = args.lam_jac * min(1.0, global_step / max(1, args.jac_warmup))
                        pi_loss = pi_loss + lam_jac_t * actor_net._jac_pen.mean()
                        last_jac_pen = _to_float(actor_net._jac_pen.mean().detach())
                    if not bool(torch.isfinite(pi_loss)):
                        raise FloatingPointError(f"step {global_step}: non-finite pi_loss")

                    actor_opt.zero_grad(set_to_none=True)
                    pi_loss.backward()

                    gn = nn.utils.clip_grad_norm_(
                        actor_net.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
                    actor_opt.step()

                    lp_ctrl = act_logp.detach()
                    if args.ent_track:
                        ent_batch = (-lp_ctrl).mean().item()
                        ent_ema = ent_batch if ent_ema is None else 0.99 * ent_ema + 0.01 * ent_batch
                        target_entropy = max(ent_ema - args.ent_margin,
                                             -float(act_dim) * args.ent_floor_scale)

                    if args.lp_winsor > 0:
                        qs = torch.quantile(lp_ctrl, torch.tensor(
                            [args.lp_winsor, 1.0 - args.lp_winsor], device=lp_ctrl.device))
                        lp_ctrl = lp_ctrl.clamp(qs[0], qs[1])
                    ctrl_target = target_entropy
                    alpha_loss = -(log_alpha * (lp_ctrl + ctrl_target)).mean()
                    if not bool(torch.isfinite(alpha_loss)):
                        raise FloatingPointError(f"step {global_step}: non-finite alpha_loss")
                    alpha_opt.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_opt.step()
                    with torch.no_grad():

                        if args.alpha_leak > 0:
                            prior = args.alpha_prior
                            log_alpha.mul_(1.0 - args.alpha_leak).add_(
                                args.alpha_leak * math.log(prior))
                        if args.alpha_min > 0:
                            log_alpha.clamp_(min=math.log(args.alpha_min))
                    alpha = log_alpha.exp().item()

                    last_pi_loss = _to_float(pi_loss)
                    last_qpi = _to_float(q_pi.mean().detach())
                    last_logp = (_to_float(act_logp.mean().detach())
                                 if act_logp is not None else None)
                    last_gnorm = _to_float(gn)

                soft_update(q1_targ, q1, args.tau)
                soft_update(q2_targ, q2, args.tau)

            if global_step - last_tb >= args.tb_interval:
                last_tb = global_step
                writer.add_scalar("loss/q_loss", last_q_loss, global_step)
                if last_pi_loss is not None:
                    writer.add_scalar("loss/pi_loss", last_pi_loss, global_step)
                    writer.add_scalar("stats/q_pi_mean", last_qpi, global_step)
                    if last_logp is not None:
                        writer.add_scalar("stats/logp_pi", last_logp, global_step)
                    if last_logp_raw_mean is not None:
                        writer.add_scalar("stats/logp_raw_mean", last_logp_raw_mean, global_step)
                        writer.add_scalar("stats/logp_raw_p05", last_logp_raw_p05, global_step)
                        writer.add_scalar("stats/logp_raw_p95", last_logp_raw_p95, global_step)
                    if last_logp_clip_frac is not None:
                        writer.add_scalar("stats/logp_clip_frac", last_logp_clip_frac, global_step)
                        writer.add_scalar("stats/logp_clip_abs_excess_mean", last_logp_clip_excess, global_step)
                if last_jac_pen is not None:
                    writer.add_scalar("loss/jac_pen", last_jac_pen, global_step)
                if auto_alpha and args.ent_track:
                    writer.add_scalar("coeff/target_entropy", target_entropy, global_step)
                writer.add_scalar("coeff/alpha", alpha, global_step)
                if last_gnorm is not None:
                    writer.add_scalar("grad/actor_grad_norm", last_gnorm, global_step)
                if getattr(actor_net, "last_fwd", None):
                    for fk, fv in actor_net.last_fwd.items():
                        if fk != "res_curve" and fv is not None:
                            writer.add_scalar(f"solver/fwd_{fk}", fv, global_step)
                for dk, dv in getattr(actor_net, "last_diag", {}).items():
                    writer.add_scalar(f"diag/{dk}", dv, global_step)
                writer.add_scalar("charts/sps", int(global_step / max(time.perf_counter() - start_time, 1e-6)), global_step)

        if not args.save_full_state and args.ckpt_interval > 0 and global_step - last_ckpt >= args.ckpt_interval:
            last_ckpt = global_step
            _save_step_checkpoint(
                {"actor": actor_net.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                 "log_alpha": log_alpha.detach().cpu(), "global_step": global_step,
                 "args": vars(args)},
                f"runs/{run_name}",
                global_step,
            )

        if eval_env is not None and global_step - last_eval >= args.eval_interval:
            last_eval = global_step
            eval_ret = eval_policy(eval_env, actor_net, device, args.eval_envs,
                                   eval_seed=args.seed + 10_000 + last_eval)
            eval_native_success = getattr(
                eval_env, "_sacflow_last_native_success_rate", None
            )
            writer.add_scalar("charts/eval_return_det", eval_ret, global_step)
            if eval_native_success is not None:
                writer.add_scalar(
                    "charts/eval_native_success_rate_det",
                    eval_native_success,
                    global_step,
                )
            print(
                "eval: | "
                f"step={global_step} eval_return_det={eval_ret:.1f} "
                f"eval_native_success_rate_det={eval_native_success}"
            )
            if args.eval_stochastic > 0:
                stoch_success, stoch_episodes, stoch_return = 0.0, 0, 0.0

                fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    for pass_index in range(args.eval_stochastic):
                        stoch_return += eval_policy(
                            eval_env, actor_net, device, args.eval_envs,
                            eval_seed=args.seed + 30_000 + last_eval + pass_index,
                            stochastic=True,
                        )
                        rate = getattr(eval_env, "_sacflow_last_native_success_rate", None)
                        if rate is not None:
                            stoch_success += rate * args.eval_envs
                            stoch_episodes += args.eval_envs
                writer.add_scalar(
                    "charts/eval_return_stoch", stoch_return / args.eval_stochastic, global_step)
                if stoch_episodes > 0:
                    writer.add_scalar(
                        "charts/eval_native_success_rate_stoch",
                        stoch_success / stoch_episodes, global_step)
                print(
                    "eval-stoch: | "
                    f"step={global_step} passes={args.eval_stochastic} "
                    f"episodes={stoch_episodes} "
                    f"native_success={stoch_success / max(stoch_episodes, 1):.4f}"
                )

        if global_step - last_log >= args.log_interval:
            last_log = global_step
            stats = summarize_finished_trajs(
                finished_trajs_window,
                act_steps=args.act_steps,
                best_reward_threshold_for_success=args.best_reward_threshold_for_success,
            )

            writer.add_scalar("charts/num_episode_finished", stats["num_episode_finished"], global_step)
            writer.add_scalar("charts/avg_episode_reward", stats["avg_episode_reward"], global_step)
            writer.add_scalar("charts/std_episode_reward", stats["std_episode_reward"], global_step)
            writer.add_scalar("charts/avg_best_reward", stats["avg_best_reward"], global_step)
            writer.add_scalar("charts/success_rate", stats["success_rate"], global_step)
            writer.add_scalar("charts/avg_episode_length", stats["avg_episode_length"], global_step)
            sps = int(global_step / max(time.perf_counter() - start_time, 1e-6))

            print("rollout:" + format_kv(dict(
                step=global_step, sps=sps, alpha=alpha, logp=last_logp,
                q_loss=last_q_loss, pi_loss=last_pi_loss, qpi=last_qpi, **stats,
            )))

            finished_trajs_window = []


        if args.save_full_state and args.ckpt_interval > 0 and global_step - last_ckpt >= args.ckpt_interval:
            last_ckpt = global_step
            _save_step_checkpoint(full_state_checkpoint(), f"runs/{run_name}", global_step)


    if args.ckpt_interval > 0 and global_step != last_ckpt:
        _save_step_checkpoint(
            (full_state_checkpoint() if args.save_full_state else
             {"actor": actor_net.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
              "log_alpha": log_alpha.detach().cpu(), "global_step": global_step,
              "args": vars(args)}),
            f"runs/{run_name}",
            global_step,
        )

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        ckpt = {
            "actor": actor_net.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "args": vars(args),
        }
        if args.save_full_state:
            ckpt = full_state_checkpoint()
        path = f"runs/{run_name}/{args.exp_name}.pt"
        torch.save(ckpt, path)
        print(f"model saved to {path}")

    env.close()
    writer.close()

if __name__ == "__main__":
    args = Args()
    main(args)
