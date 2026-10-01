"""Focused checks for the isolated narrow-navigation docking runtime."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
NARROW = ROOT / "src/docking_narrow"
FULL = ROOT / "src/docking_full"
sys.path.insert(0, str(NARROW))
import multigoal_docking_env as environment
import run_multigoal_docking_narrow as launcher
import evaluate_multigoal_docking_narrow as evaluator


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def original_environment():
    spec = importlib.util.spec_from_file_location(
        "original_docking_geometry", FULL / "multigoal_docking_env.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(gym, "register"):
        spec.loader.exec_module(module)
    return module.MultiGoalDockingEnv


@pytest.mark.parametrize("goal,direction", [
    (0, [1, 0]), (1, [-1, 0]), (2, [0, 1]), (3, [0, -1]),
])
def test_four_rotated_ideal_routes(goal, direction):
    env = gym.make(launcher.ENVIRONMENT)
    try:
        obs, _ = env.reset(seed=7, options={"fixed_origin": True})
        assert obs.tolist() == [0, 0, 0, 0, 0]
        assert env.unwrapped.navigation_width == 0.20
        assert env.unwrapped.docking_sigma == 0.03
        rewards = []
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(direction)
            rewards.append(reward)
            assert not terminated and not truncated
            assert info["goal_reached"] == float(step == 4)
        assert rewards == [-4.0, -3.0, -2.0, -1.0, 0.0]
        assert info["selected_goal"] == goal
        np.testing.assert_array_equal(obs[3:], env.unwrapped.goals[goal])
        assert obs[2] == 1
        _, reward, terminated, truncated, info = env.step(
            env.unwrapped.docking_targets[goal])
        assert reward == 10 and terminated and not truncated
        assert info["success"] == 1 and info["docking_distance"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("goal", range(4))
def test_transverse_miss_can_be_corrected(goal):
    env = environment.MultiGoalDockingEnv(docking_sigma=0.03, navigation_width=0.20)
    env.reset(options={"fixed_origin": True})
    direction = env.goals[goal] / 5
    transverse = np.array([-direction[1], direction[0]], np.float32)
    env.pos = env.goals[goal] + 0.25 * transverse
    _, reward, _, truncated, info = env.step([0, 0])
    assert info["goal_reached"] == 0 and not truncated
    assert reward == pytest.approx(-0.25)
    _, reward, _, truncated, info = env.step(-0.10 * transverse)
    assert info["goal_reached"] == 1 and info["selected_goal"] == goal
    assert reward == pytest.approx(-0.15) and not truncated
    env.close()


@pytest.mark.parametrize("offset,reached", [
    ([1.0, 0.0], False), ([0.999, 0.0], True),
    ([0.0, 0.199], True), ([0.0, 0.201], False),
    ([0.6, 0.159], True), ([0.6, 0.161], False),
])
def test_ellipse_boundary_and_combined_axes(offset, reached):
    env = environment.MultiGoalDockingEnv(docking_sigma=0.03, navigation_width=0.20)
    env.reset(options={"fixed_origin": True})
    env.pos = env.goals[0] + np.asarray(offset, np.float32)
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert bool(info["goal_reached"]) is reached
    assert not terminated and not truncated
    assert reward == pytest.approx(-np.linalg.norm(offset), abs=1e-6)
    env.close()


def test_timeout_and_last_step_arrival():
    env = environment.MultiGoalDockingEnv(docking_sigma=0.03, navigation_width=0.20)
    env.reset(options={"fixed_origin": True})
    for _ in range(30):
        _, _, terminated, truncated, info = env.step([0, 0])
    assert truncated and not terminated and info["selected_goal"] == -1
    env.reset(options={"fixed_origin": True})
    for _ in range(25):
        env.step([0, 0])
    for _ in range(5):
        _, _, terminated, truncated, info = env.step([1, 0])
    assert not terminated and not truncated and info["goal_reached"] == 1
    _, _, terminated, truncated, info = env.step(env.docking_targets[0])
    assert terminated and not truncated and info["success"] == 1
    env.close()


@pytest.mark.parametrize("width", [0, -0.1, 1.01, float("nan")])
def test_invalid_width(width):
    with pytest.raises(ValueError, match="navigation_width"):
        environment.MultiGoalDockingEnv(navigation_width=width)


@pytest.mark.parametrize("sigma", [0.03, 0.06])
def test_width_one_matches_original_episode_transitions(sigma):
    reference = original_environment()(docking_sigma=sigma)
    current = environment.MultiGoalDockingEnv(docking_sigma=sigma, navigation_width=1)
    rng = np.random.default_rng(34)
    for seed in range(12):
        options = {"fixed_origin": seed % 2 == 0}
        old_obs, old_info = reference.reset(seed=seed, options=options)
        new_obs, new_info = current.reset(seed=seed, options=options)
        np.testing.assert_array_equal(old_obs, new_obs)
        assert old_info == new_info
        for step in range(31):
            action = rng.uniform(-1.5, 1.5, 2) if seed % 3 else [1, 0]
            old = reference.step(action)
            new = current.step(action)
            np.testing.assert_array_equal(old[0], new[0])
            assert old[1:] == new[1:]
            if old[2] or old[3]:
                break
    reference.close()
    current.close()


@pytest.mark.parametrize("method", launcher.METHODS)
def test_only_environment_and_run_names_change_from_tight_recipe(method):
    filename = "run_multigoal_docking_full.py"
    assert (NARROW / filename).read_bytes() == (FULL / filename).read_bytes()
    old = asdict(launcher.base.configure(method, "0.03", 1, 200_000))
    new = asdict(launcher.configure(method, "0.03", 1, 200_000))
    differences = {key for key in old if old[key] != new[key]}
    assert differences == {"run_name", "exp_name", "env_id"}
    args, contract = launcher.run_contract(method, 1)
    assert contract["configuration"] == asdict(args)
    assert contract["geometry"] == {
        "navigation_parallel_radius": 1.0, "navigation_width": 0.20,
        "docking_sigma": 0.03, "docking_tolerance": 0.06,
    }


def test_matched_integrator_control():
    explicit = asdict(launcher.configure("explicit", "0.03", 1, 200_000))
    implicit = asdict(launcher.configure("deflow", "0.03", 1, 200_000))
    assert {key for key in explicit if explicit[key] != implicit[key]} == {
        "run_name", "exp_name", "integration"}
    assert explicit["integration"] == "explicit"
    assert implicit["integration"] == "implicit"


def write_checkpoint(tmp_path, method):
    args, contract = launcher.run_contract(method, 1, smoke=True)
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
def test_strict_checkpoint_reconstruction_and_geometry(method, tmp_path):
    checkpoint = write_checkpoint(tmp_path, method)
    result = evaluator.evaluate(checkpoint, n=2)
    assert result == evaluator.evaluate(checkpoint, n=2)
    assert result["method"] == method and result["checkpoint_global_step"] == 2048
    assert result["environment"] == launcher.ENVIRONMENT
    assert result["geometry"] == launcher.GEOMETRY
    rows, metrics = result["episodes"], result["metrics"]
    assert len(rows) == 2
    assert sum(metrics["p_i"]) == pytest.approx(metrics["total_success"])
    assert metrics["arrival_rate"] + metrics["timeout_rate"] == 1
    assert metrics["mean_navigation_steps"] == np.mean([r["navigation_steps"] for r in rows])
    assert metrics["arrivals_by_goal"] == [sum(r["goal"] == i for r in rows) for i in range(4)]

    payload = torch.load(checkpoint, weights_only=False)
    payload["actor"]["unexpected_parameter"] = torch.zeros(1)
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        evaluator.evaluate(checkpoint, n=1)

    contract_path = tmp_path / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["geometry"]["navigation_width"] = 0.10
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="geometry differs"):
        evaluator.evaluate(checkpoint, n=1)


@pytest.mark.parametrize("method", launcher.METHODS)
def test_tiny_cpu_optimizer_smoke(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = launcher.engine
    original_buffer = engine.replay_buffer
    observed = {"samples": 0, "checkpoint": None}

    class CountingBuffer(original_buffer):
        def sample(self, *args, **kwargs):
            observed["samples"] += 1
            return super().sample(*args, **kwargs)

    monkeypatch.setattr(engine, "replay_buffer", CountingBuffer)
    monkeypatch.setattr(engine, "_save_step_checkpoint", lambda state, *args:
                        observed.update(checkpoint=state))
    args = launcher.configure(method, "0.03", 1, 16, smoke=True)
    args.learning_starts, args.batch_size, args.buffer_size = 8, 8, 64
    args.eval_interval, args.ckpt_interval, args.save_model = 0, 16, False
    engine.main(args)
    assert observed["samples"] > 0
    state = observed["checkpoint"]
    assert state["global_step"] == 16
    for component in ("actor", "q1", "q2"):
        assert all(torch.isfinite(tensor).all() for tensor in state[component].values())
    assert torch.isfinite(state["log_alpha"]).all()
