"""Q-score-matching diffusion policy and training loop."""

import math
import os
import random
import time
import uuid
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import sac_anderson_flow as base
from sac_flow_parent import parent_critic


def vp_beta_schedule(T: int) -> torch.Tensor:

    t = torch.arange(1, T + 1, dtype=torch.float64)
    b_max, b_min = 10.0, 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    return (1.0 - alpha).to(torch.float32)


def mish_mlp(sizes):


    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.Mish())
    return nn.Sequential(*layers)


class qsm_actor(nn.Module):


    def __init__(self, obs_dim, act_dim, act_low, act_high, *,
                 T: int = 5, time_dim: int = 64, hidden: int = 512, M_q: float = 50.0,
                 ddpm_temperature: float = 1.0, clip_sampler: bool = True):
        super().__init__()
        self.act_dim = act_dim
        self.T = T
        self.M_q = M_q
        self.ddpm_temperature = ddpm_temperature
        self.clip_sampler = clip_sampler

        self.time_w = nn.Parameter(torch.randn(time_dim // 2, 1) * 0.2)
        self.cond_mlp = mish_mlp([time_dim, 128, 128])
        self.reverse_mlp = mish_mlp([act_dim + obs_dim + 128, hidden, hidden, act_dim])

        betas = vp_beta_schedule(T)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hats", torch.cumprod(alphas, dim=0))

        act_low = torch.as_tensor(act_low, dtype=torch.float32)
        act_high = torch.as_tensor(act_high, dtype=torch.float32)
        self.register_buffer("action_scale", (act_high - act_low) / 2.0)
        self.register_buffer("action_bias", (act_high + act_low) / 2.0)

    def _time_features(self, t):
        f = 2.0 * math.pi * (t @ self.time_w.t())
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)

    def eps(self, obs, a, t):

        cond = self.cond_mlp(self._time_features(t))
        return self.reverse_mlp(torch.cat([a, obs, cond], dim=-1))

    @torch.no_grad()
    def sample(self, obs):

        B = obs.shape[0]
        x = torch.randn(B, self.act_dim, device=obs.device, dtype=obs.dtype)
        for time_step in range(self.T - 1, -1, -1):
            t_in = torch.full((B, 1), float(time_step), device=obs.device, dtype=obs.dtype)
            eps_pred = self.eps(obs, x, t_in)
            x = (x - (1.0 - self.alphas[time_step]) / torch.sqrt(1.0 - self.alpha_hats[time_step])
                 * eps_pred) / torch.sqrt(self.alphas[time_step])
            if time_step > 0:
                x = x + torch.sqrt(self.betas[time_step]) * self.ddpm_temperature * torch.randn_like(x)
            if self.clip_sampler:
                x = x.clamp(-1.0, 1.0)
        return x.clamp(-1.0, 1.0)

    def act(self, obs, deterministic: bool = False):


        a = self.sample(obs)
        if not deterministic:
            a = (a + 0.1 * torch.randn_like(a)).clamp(-1.0, 1.0)
        return a * self.action_scale + self.action_bias

    def forward(self, obs):
        return self.act(obs, deterministic=False)


def qsm_actor_loss(actor, q1, q2, obs, act_norm):


    t = torch.randint(0, actor.T, (act_norm.shape[0],), device=obs.device)
    noise = torch.randn_like(act_norm)
    ah = actor.alpha_hats[t].unsqueeze(-1)
    noisy = torch.sqrt(ah) * act_norm + torch.sqrt(1.0 - ah) * noise

    def q_grad(qnet):
        leaf = noisy.detach().requires_grad_(True)
        q = qnet(obs, leaf * actor.action_scale + actor.action_bias)
        return torch.autograd.grad(q.sum(), leaf)[0]

    g = 0.5 * (q_grad(q1) + q_grad(q2))
    eps_pred = actor.eps(obs, noisy, t.unsqueeze(-1).to(obs.dtype))
    loss = ((-actor.M_q * g - eps_pred) ** 2).mean()
    return loss, g, eps_pred


def qsm_critic_target(actor, q1_targ, q2_targ, next_obs, rew, done, gamma):


    with torch.no_grad():
        na = actor.sample(next_obs)
        na = (na + 0.1 * torch.randn_like(na)).clamp(-1.0, 1.0)
        env_na = na * actor.action_scale + actor.action_bias
        q_next = torch.min(q1_targ(next_obs, env_na), q2_targ(next_obs, env_na))
        return rew + (1.0 - done) * gamma * q_next


@dataclass
class QSMArgs(base.Args):
    denoising_steps: int = 5
    M_q: float = 50.0
    lr_decay_steps: int = 2_000_000
    campaign_artifacts: bool = False
    max_q_loss: float = 1e12


def main(args: QSMArgs):
    if args.campaign_artifacts and (args.resume_ckpt or args.resume_full_state or args.save_full_state):
        raise ValueError("QSM campaign is fresh-only; exact replay/simulator resume is not implemented")
    run_name = (args.run_name if args.campaign_artifacts else "") or (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4().hex[:12]}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = base.select_device(args, run_name)
    writer = SummaryWriter(f"runs/{run_name}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    env = base.make_vec_env(args.env_id, args.seed, args.num_envs, args.capture_video, run_name)
    single_act_space = env.single_action_space
    assert isinstance(single_act_space, gym.spaces.Box), "only continuous action space is supported"

    obs_dim = int(np.prod(env.single_observation_space.shape))
    act_dim = int(np.prod(single_act_space.shape))
    act_low = single_act_space.low.astype(np.float32)
    act_high = single_act_space.high.astype(np.float32)

    actor_net = qsm_actor(obs_dim, act_dim, act_low, act_high,
                          T=args.denoising_steps, M_q=args.M_q).to(device)
    q1 = parent_critic(obs_dim, act_dim).to(device)
    q2 = parent_critic(obs_dim, act_dim).to(device)
    q1_targ = parent_critic(obs_dim, act_dim).to(device)
    q2_targ = parent_critic(obs_dim, act_dim).to(device)
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())


    n_q1 = sum(p.numel() for p in q1.parameters())
    print("identity-probe: | "
          f"actor={type(actor_net).__module__}.{type(actor_net).__name__} "
          f"critic={type(q1).__module__}.{type(q1).__name__} "
          f"critic_params={n_q1} "
          "entropy_channel=ABSENT-BY-DESIGN")
    assert type(actor_net).__module__ == "qsm_actor", "actor class substituted!"
    assert type(q1).__name__ == "parent_critic", "critic class substituted!"

    actor_opt = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)


    decay = args.lr_decay_steps
    actor_sched = torch.optim.lr_scheduler.LambdaLR(
        actor_opt, lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, decay) / decay)))
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.critic_lr)

    rb = base.replay_buffer(obs_dim, act_dim, args.buffer_size, device)

    eval_env = None
    last_eval = 0
    if args.eval_interval > 0:
        eval_env = base.make_vec_env(args.env_id, args.seed + 10_000, args.eval_envs, False, run_name)

    obs, info = env.reset(seed=args.seed)
    obs = obs.astype(np.float32).reshape(args.num_envs, -1)

    cur_rew_trajs = [[] for _ in range(args.num_envs)]
    finished_trajs_window = []
    prev_autoreset = np.zeros(args.num_envs, dtype=bool)

    global_step = 0
    n_updates = 0
    last_log = 0
    last_tb = 0
    last_ckpt = 0
    last_q1 = last_q2 = last_q_loss = None
    last_score_loss = last_gq_norm = last_eps_norm = None
    start_time = time.perf_counter()

    def save_campaign_checkpoint():


        state = {
            "actor": actor_net.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
            "q1_targ": q1_targ.state_dict(), "q2_targ": q2_targ.state_dict(),
            "opt_states": {"actor_opt": actor_opt.state_dict(), "q_opt": q_opt.state_dict()},
            "actor_scheduler": actor_sched.state_dict(), "rng": base._rng_state(),
            "global_step": global_step, "n_updates": n_updates, "args": vars(args),
            "run_id": run_name, "actor_class": "qsm_actor.qsm_actor",
            "actor_constructor": {"T": actor_net.T, "M_q": actor_net.M_q,
                                  "time_dim": 64, "hidden": 512,
                                  "ddpm_temperature": 1.0, "clip_sampler": True},
            "full_state": False, "resume_supported": False,
            "evaluation_policy": "ancestral DDPM; no extra behavior noise; stochastic",
        }
        for name in ("actor", "q1", "q2", "q1_targ", "q2_targ"):
            if any(not torch.isfinite(value).all() for value in state[name].values()):
                raise FloatingPointError(f"nonfinite {name} at step {global_step}")
        base._save_step_checkpoint(state, f"runs/{run_name}", global_step)

    while global_step < args.total_timesteps:
        if global_step < args.learning_starts:
            act = env.action_space.sample().astype(np.float32)
        else:
            with torch.no_grad():
                act_t = actor_net(torch.as_tensor(obs, device=device)).cpu().numpy()
            act = np.clip(act_t, act_low, act_high).astype(np.float32)

        next_obs, rew, term, trunc, info = env.step(act)
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

        if global_step >= args.learning_starts and len(rb) >= args.batch_size:
            num_updates = max(1, int(round(args.num_envs * args.utd_ratio)))
            for _ in range(num_updates):
                b_obs, b_act, b_next_obs, b_rew, b_done = rb.sample(args.batch_size)

                target_q = qsm_critic_target(
                    actor_net, q1_targ, q2_targ, b_next_obs, b_rew, b_done, args.gamma)
                q1_pred = q1(b_obs, b_act)
                q2_pred = q2(b_obs, b_act)
                q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
                if args.campaign_artifacts:
                    value = float(q_loss.detach())
                    if not math.isfinite(value) or value > args.max_q_loss:
                        raise FloatingPointError(f"step {global_step}: q_loss={value}")

                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                q_opt.step()

                n_updates += 1
                last_q1 = base._to_float(q1_pred.mean())
                last_q2 = base._to_float(q2_pred.mean())
                last_q_loss = base._to_float(q_loss)

                if n_updates % args.policy_frequency == 0:
                    b_act_norm = (b_act - actor_net.action_bias) / actor_net.action_scale
                    score_loss, g, eps_pred = qsm_actor_loss(actor_net, q1, q2, b_obs, b_act_norm)
                    if args.campaign_artifacts and not torch.isfinite(score_loss):
                        raise FloatingPointError(f"step {global_step}: nonfinite QSM score loss")

                    actor_opt.zero_grad(set_to_none=True)
                    score_loss.backward()
                    actor_opt.step()
                    actor_sched.step()

                    last_score_loss = base._to_float(score_loss)
                    last_gq_norm = base._to_float(g.norm(dim=-1).mean())
                    last_eps_norm = base._to_float(eps_pred.detach().norm(dim=-1).mean())

                base.soft_update(q1_targ, q1, args.tau)
                base.soft_update(q2_targ, q2, args.tau)

            if global_step - last_tb >= args.tb_interval:
                last_tb = global_step
                writer.add_scalar("loss/q_loss", last_q_loss, global_step)
                if last_score_loss is not None:
                    writer.add_scalar("loss/qsm_score_loss", last_score_loss, global_step)
                    writer.add_scalar("stats/grad_q_norm", last_gq_norm, global_step)
                    writer.add_scalar("stats/eps_pred_norm", last_eps_norm, global_step)
                    writer.add_scalar("stats/actor_lr", actor_sched.get_last_lr()[0], global_step)
                writer.add_scalar("charts/sps", int(global_step / max(time.perf_counter() - start_time, 1e-6)), global_step)

        if args.ckpt_interval > 0 and global_step - last_ckpt >= args.ckpt_interval:
            last_ckpt = global_step
            if args.campaign_artifacts:
                save_campaign_checkpoint()
            else:
                os.makedirs(f"runs/{run_name}", exist_ok=True)
                torch.save(
                    {"actor": actor_net.state_dict(), "q1": q1.state_dict(), "q2": q2.state_dict(),
                     "global_step": global_step},
                    f"runs/{run_name}/ckpt.pt",
                )

        if eval_env is not None and global_step - last_eval >= args.eval_interval:
            last_eval = global_step
            eval_ret = base.eval_policy(eval_env, actor_net, device, args.eval_envs,
                                        eval_seed=args.seed + 10_000 + last_eval)
            suffix = "qsm_policy" if args.campaign_artifacts else "det"
            writer.add_scalar(f"charts/eval_return_{suffix}", eval_ret, global_step)
            if args.campaign_artifacts:
                success = getattr(eval_env, "_sacflow_last_native_success_rate", None)
                if success is None:
                    raise RuntimeError("QSM evaluation environment has no native success signal")
                writer.add_scalar("charts/eval_native_success_rate_qsm_policy", success, global_step)
            print(f"eval: | step={global_step} eval_return_{suffix}={eval_ret:.1f}")

        if global_step - last_log >= args.log_interval:
            last_log = global_step
            stats = base.summarize_finished_trajs(
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

            print("rollout:" + base.format_kv(dict(
                step=global_step, sps=sps, q1=last_q1, q2=last_q2, q_loss=last_q_loss,
                score_loss=last_score_loss, gq_norm=last_gq_norm, eps_norm=last_eps_norm,
                **stats,
            )))

            finished_trajs_window = []

    if args.campaign_artifacts and args.save_model:
        if last_ckpt != global_step:
            save_campaign_checkpoint()
    elif args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        ckpt = {
            "actor": actor_net.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "args": vars(args),
        }
        path = f"runs/{run_name}/{args.exp_name}.pt"
        torch.save(ckpt, path)
        print(f"model saved to {path}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main(QSMArgs())
