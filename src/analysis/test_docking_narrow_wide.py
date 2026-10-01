"""Focused checks for narrow navigation with loose docking tolerance."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/docking_narrow_wide"))
import multigoal_docking_env as environment
import run_multigoal_docking_narrow_wide as launcher
import evaluate_multigoal_docking_narrow_wide as evaluator


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_gymnasium_contract():
    env = gym.make(launcher.ENVIRONMENT).unwrapped
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("goal", range(4))
@pytest.mark.parametrize("offset,success", [(0, 1), (.09, 1), (.121, 0)])
def test_rotated_routes_and_docking_tolerance(goal, offset, success):
    env = gym.make(launcher.ENVIRONMENT)
    env.reset(seed=7, options={"fixed_origin": True})
    direction = env.unwrapped.goals[goal] / 5
    rewards = []
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(direction)
        rewards.append(reward)
        assert not terminated and not truncated
        assert env.observation_space.contains(obs)
    assert rewards == [-4., -3., -2., -1., 0.]
    assert info["selected_goal"] == goal and info["goal_reached"] == 1
    target = env.unwrapped.docking_targets[goal]
    _, reward, terminated, truncated, info = env.step(target + offset * direction)
    assert terminated and not truncated and info["success"] == success
    assert reward == pytest.approx(10 * success - offset, abs=1e-6)
    with pytest.raises(RuntimeError, match="episode ended"):
        env.step(target)
    env.close()


def test_only_docking_tolerance_changes_episode_dynamics():
    for seed in range(20):
        tight = environment.MultiGoalDockingEnv(docking_sigma=.03, navigation_width=.2)
        wide = environment.MultiGoalDockingEnv(docking_sigma=.06, navigation_width=.2)
        np.testing.assert_array_equal(tight.reset(seed=seed)[0], wide.reset(seed=seed)[0])
        rng = np.random.default_rng(seed)
        for step in range(31):
            action = rng.uniform(-1, 1, 2) if seed % 2 else [1, 0]
            before = tight.phase
            a, b = tight.step(action), wide.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[2:4] == b[2:4]
            if before == "navigate":
                assert a[1:] == b[1:]
            else:
                assert b[-1]["success"] >= a[-1]["success"]
                assert b[1] - a[1] == 10 * (b[-1]["success"] - a[-1]["success"])
            if a[2] or a[3]:
                break
        tight.close()
        wide.close()


def write_checkpoint(tmp_path, method):
    args, contract = launcher.run_contract(method, 1, smoke=True)
    if method == "qsm":
        from qsm_actor import qsm_actor
        actor = qsm_actor(5, 2, -np.ones(2), np.ones(2), T=args.denoising_steps, M_q=args.M_q)
    else:
        actor = launcher.engine.deq_multistep_flow_actor(
            5, 2, -np.ones(2, np.float32), np.ones(2, np.float32),
            denoising_steps=args.denoising_steps, integration=args.integration,
            density_estimator=args.density_estimator,
            density_diag_every=args.density_diag_every,
            density_diag_batch=args.density_diag_batch,

            gate_bias=args.gate_bias,
            jac_sigma_target=args.jac_sigma_target if args.lam_jac > 0 else 0.0,
            hidden_dim=128, u_scale=args.u_scale,
            max_iter_fwd=15, tol_fwd=1e-3, lam_fwd=1e-2, beta_fwd=0.7,
            max_iter_bwd=10, tol_bwd=1e-3, lam_bwd=1e-2, beta_bwd=0.7,
        )
    checkpoint = tmp_path / "ckpt_step_000002048.pt"
    torch.save({"args": asdict(args), "actor": actor.state_dict(), "global_step": 2048},
               checkpoint)
    (tmp_path / "run_contract.json").write_text(json.dumps(contract))
    return checkpoint


@pytest.mark.parametrize("method", launcher.METHODS)
def test_checkpoint_reconstructs_and_repeats(method, tmp_path):
    checkpoint = write_checkpoint(tmp_path, method)
    result = evaluator.evaluate(checkpoint, n=2)
    assert result == evaluator.evaluate(checkpoint, n=2)
    assert result["geometry"] == launcher.GEOMETRY
    assert result["method"] == method and result["n_episodes"] == 2
    assert result["metrics"]["arrival_rate"] + result["metrics"]["timeout_rate"] == 1


def test_method_identity_survives_import_order():
    for method in list(launcher.METHODS) + list(reversed(launcher.METHODS)):
        _, contract = launcher.run_contract(method, 1)
        assert contract["critic"] == "sac_flow_parent.parent_critic"
        if method in ("deflow", "explicit"):
            assert contract["actor"].endswith("GatedDockingActor")
            assert launcher.engine.deq_multistep_flow_actor.__name__ == "GatedDockingActor"
