"""Focused checks for the isolated full docking runtime."""

import json
from pathlib import Path
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch

FULL = Path(__file__).resolve().parents[2] / "src/docking_full"
sys.path.insert(0, str(FULL))
import multigoal_docking_env
from step_accounting import accrue_updates, real_transitions
import run_multigoal_docking_full as launcher
import evaluate_multigoal_docking_full as evaluator


class OneStepEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(5, np.float32), {}

    def step(self, action):
        return np.zeros(5, np.float32), 1.0, True, False, {"success": 1.0}


def test_next_step_reset_slots_and_fractional_utd():
    env = gym.vector.SyncVectorEnv([lambda: gym.make("MultiGoalDock-v0") for _ in range(2)])
    try:
        env.reset(seed=5)

        env.envs[0].unwrapped.pos = np.array([5, 0], dtype=np.float32)
        env.envs[0].unwrapped.phase = "dock"
        env.envs[0].unwrapped.selected_goal = 0
        _, _, terminated, truncated, _ = env.step(np.array([[0.6, 0.2], [0, 0]], np.float32))
        previous_autoreset = terminated | truncated
        assert previous_autoreset.tolist() == [True, False]
        assert real_transitions(previous_autoreset) == 1
        _, reward, terminated, truncated, _ = env.step(np.zeros((2, 2), np.float32))
        assert reward[0] == 0 and not terminated[0] and not truncated[0]
        n_real = real_transitions(previous_autoreset)
        assert n_real == 1
        updates, credit = accrue_updates(0.0, n_real, 0.5)
        assert updates == 0 and credit == 0.5
        updates, credit = accrue_updates(credit, 1, 0.5)
        assert updates == 1 and credit == 0.0
        assert accrue_updates(credit, 0, 0.5) == (0, 0.0)
    finally:
        env.close()


@pytest.mark.parametrize("method", ["sac", "qsm"])
def test_fresh_loop_counts_replay_inserts_not_reset_slots(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = launcher.engine
    original_buffer = engine.replay_buffer
    observed = {"inserts": 0, "step": None}

    class CountingBuffer(original_buffer):
        def add(self, *args, **kwargs):
            observed["inserts"] += 1
            return super().add(*args, **kwargs)

    monkeypatch.setattr(engine, "replay_buffer", CountingBuffer)
    monkeypatch.setattr(engine, "make_vec_env", lambda *args, **kwargs: gym.vector.SyncVectorEnv(
        [lambda: OneStepEnv() for _ in range(8)]))
    monkeypatch.setattr(engine, "_save_step_checkpoint", lambda state, *args: observed.update(
        step=state["global_step"]))
    args = launcher.configure(method, "0.06", 1, 128, smoke=True)
    args.learning_starts = 1024
    args.eval_interval = 0
    args.ckpt_interval = 128
    args.save_model = False
    if method == "qsm":
        import qsm_actor

        qsm_actor.main(args)
    else:
        engine.main(args)
    assert observed == {"inserts": 128, "step": 128}


@pytest.mark.parametrize("method", ["sac", "qsm"])
def test_utd_credit_tracks_real_inserts(method, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = launcher.engine
    original_buffer = engine.replay_buffer
    observed = {"inserts": 0, "samples": 0}

    class CountingBuffer(original_buffer):
        def add(self, *args, **kwargs):
            observed["inserts"] += 1
            return super().add(*args, **kwargs)

        def sample(self, *args, **kwargs):
            observed["samples"] += 1
            return super().sample(*args, **kwargs)

    monkeypatch.setattr(engine, "replay_buffer", CountingBuffer)
    monkeypatch.setattr(engine, "make_vec_env", lambda *args, **kwargs: gym.vector.SyncVectorEnv(
        [lambda: OneStepEnv() for _ in range(8)]))
    args = launcher.configure(method, "0.06", 1, 16, smoke=True)
    args.learning_starts = 8
    args.batch_size = 8
    args.buffer_size = 64
    args.utd_ratio = 0.5
    args.eval_interval = args.ckpt_interval = 0
    args.save_model = False
    if method == "qsm":
        import qsm_actor

        qsm_actor.main(args)
    else:
        engine.main(args)
    assert observed == {"inserts": 16, "samples": 8}


@pytest.mark.parametrize("method", launcher.METHODS)
def test_actor_reconstruction_and_source_contract(method, tmp_path):
    args = launcher.configure(method, "0.06", 1, 2048, smoke=True)
    low, high = -np.ones(2, np.float32), np.ones(2, np.float32)
    if method == "qsm":
        from qsm_actor import qsm_actor

        actor = qsm_actor(5, 2, low, high, T=args.denoising_steps, M_q=args.M_q)
        constructor = {"T": actor.T, "M_q": actor.M_q, "time_dim": 64,
                       "hidden": 512, "ddpm_temperature": 1.0, "clip_sampler": True}
    else:
        actor = launcher.engine.deq_multistep_flow_actor(
            5, 2, low, high,
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
        constructor = None
    run_dir = tmp_path / args.run_name
    run_dir.mkdir()
    checkpoint = run_dir / "ckpt_step_000002048.pt"
    payload = {"args": vars(args), "actor": actor.state_dict(), "global_step": 2048}
    if constructor is not None:
        payload.update(actor_class="qsm_actor.qsm_actor", actor_constructor=constructor)
    torch.save(payload, checkpoint)
    (run_dir / "run_contract.json").write_text(json.dumps({
        "run_id": args.run_name, "configuration": vars(args),
        "source_sha256": launcher.source_hashes(),
    }))
    result = evaluator.evaluate(checkpoint, n=2)
    assert result["method"] == method
    assert result["checkpoint_global_step"] == 2048
    assert len(result["episodes"]) == 2
    assert sum(result["metrics"]["p_i"]) == pytest.approx(result["metrics"]["total_success"])
    if method == "sac":
        contract_path = run_dir / "run_contract.json"
        contract = json.loads(contract_path.read_text())
        contract["source_sha256"]["step_accounting.py"] = "0" * 64
        contract_path.write_text(json.dumps(contract))
        with pytest.raises(ValueError, match="source differs"):
            evaluator.evaluate(checkpoint, n=1)
