"""Audit supplied failure logs and probe a frozen docking checkpoint."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch.func import jacrev, vmap
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/docking_full"))
import evaluate_multigoal_docking_full as evaluator

TAGS = ("grad/actor_grad_norm", "diag/logdet_step_min", "diag/negdet_frac",
        "diag/logp_raw_nonfinite_frac", "diag/logdet_clamp_frac", "loss/q_loss")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histories(manifest):
    rows = []
    for record in json.loads(manifest.read_text())["records"]:
        run = record["id"]
        log, event, terminal = [manifest.parent / record[k] for k in ("log", "events", "terminal")]
        failure, = re.findall(r"FloatingPointError: step (\d+): (.+)", log.read_text())
        assert json.loads(terminal.read_text())["exit_code"] == 1
        accumulator = EventAccumulator(str(event), size_guidance={"scalars": 0})
        accumulator.Reload()
        scalars = {tag: [{"step": e.step, "value": e.value}
                         for e in accumulator.Scalars(tag)] for tag in TAGS}
        assert all(math.isfinite(p["value"]) for points in scalars.values() for p in points)
        rows.append({"run": run, "failure_step": int(failure[0]), "reason": failure[1],
                     "last_logged_step": scalars[TAGS[0]][-1]["step"], "scalars": scalars,
                     "sources": {str(p): digest(p)
                                 for p in (log, event, terminal)}})
    return rows


def step_matrix(actor, obs, z):
    def field(u):
        return actor.vf(u[None], obs[None], u.new_zeros(1, 1))[0]
    jacobian = vmap(jacrev(field))(z)

    matrix = actor.dt * jacobian
    idx = torch.arange(actor.act_dim)
    matrix[:, idx, idx] += 1.0 + actor.logdet_eps
    return matrix


def checkpoint_probe(checkpoint, episode_seed, latent_seed, episodes, samples_per_state):
    actor, method, _, _, env_id, step, _ = evaluator.load_actor(checkpoint, torch.device("cpu"))
    assert method == "explicit"
    contract = checkpoint.parent / "run_contract.json"
    assert json.loads(contract.read_text())["source_sha256"] == evaluator.launcher.source_hashes()
    torch.manual_seed(episode_seed)
    env = evaluator.gym.make(env_id)
    docking = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=episode_seed + episode, options={"fixed_origin": True})
            while True:
                if obs[2]:
                    docking.append(torch.tensor(obs.copy()))
                with torch.no_grad():
                    action = actor.act(torch.tensor(obs)[None]).numpy()[0]
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
    torch.manual_seed(latent_seed)
    samples = actor._sample_base_z(samples_per_state, torch.device("cpu"), torch.float32)
    chosen = None
    sampled = []
    for i, obs in enumerate(docking):
        matrix = step_matrix(actor, obs, samples).detach()
        det = torch.linalg.det(matrix)
        sv = torch.linalg.svdvals(matrix)
        sampled.append({"state_index": i, "negative_determinants": int((det < 0).sum()),
                        "min_singular_value": sv[:, -1].min().item()})
        if chosen is None and (det < 0).any() and (det > 0).any():
            chosen = (obs, samples[torch.where(det > 0)[0][0]],
                      samples[torch.where(det < 0)[0][0]], i)
    assert chosen is not None, "No first-step sign-change bracket in the fixed probe bank"
    obs, positive, negative, state_index = chosen
    actor64 = copy.deepcopy(actor).double()
    left, right = positive.double(), negative.double()
    original = (left.clone(), right.clone())
    endpoint_dets = torch.linalg.det(step_matrix(actor64, obs.double(), torch.stack(original))).detach()
    assert endpoint_dets[0] > 0 and endpoint_dets[1] < 0
    for _ in range(45):
        midpoint = (left + right) / 2
        determinant = torch.linalg.det(step_matrix(actor64, obs.double(), midpoint[None])).item()
        if determinant > 0:
            left = midpoint
        else:
            right = midpoint
    center = (left + right) / 2
    guarded = step_matrix(actor64, obs.double(), center[None]).detach()[0]
    dt_jacobian = guarded - (1 + actor.logdet_eps) * torch.eye(2, dtype=torch.float64)
    eigenvalues = torch.linalg.eigvals(dt_jacobian)
    implicit_matrix = torch.eye(2, dtype=torch.float64) - dt_jacobian
    direction = original[1] - original[0]
    direction /= direction.norm()
    sensitivity = []
    for offset in (1e-2, 1e-4, 1e-6, 1e-8):
        z = (center + offset * direction)[None]
        matrix = step_matrix(actor64, obs.double(), z)
        sign, ld = torch.linalg.slogdet(matrix)
        gradients = torch.autograd.grad(-ld.clamp(-50, 50).sum(),
                                        tuple(actor64.parameters()), allow_unused=True)
        grad_norm = sum(g.square().sum() for g in gradients if g is not None).sqrt()
        sv = torch.linalg.svdvals(matrix.detach())[0]
        sensitivity.append({"offset": offset, "det_sign": sign.item(),
                            "logabsdet": ld.item(), "sigma_min": sv[-1].item(),
                            "density_step_parameter_grad_norm": grad_norm.item()})
    return {"checkpoint": str(checkpoint), "sha256": digest(checkpoint),
            "step": step, "torch_version": torch.__version__, "device": "cpu",
            "episode_seed": episode_seed, "latent_seed": latent_seed,
            "n_docking_states": len(docking), "latents_per_state": len(samples),
            "sampled_first_step": sampled, "selected_state_index": state_index,
            "observation": obs.tolist(), "bracket": [p.tolist() for p in original],
            "endpoint_determinants_float64": endpoint_dets.tolist(),
            "pole_latent_float64": center.tolist(), "bracket_width": (right-left).norm().item(),
            "pole_dtJ_eigenvalues": [[v.real.item(), v.imag.item()] for v in eigenvalues],
            "pole_guarded_sigma_min": torch.linalg.svdvals(guarded)[-1].item(),
            "same_local_J_implicit_matrix_sigma_min": torch.linalg.svdvals(implicit_matrix)[-1].item(),
            "sensitivity_float64": sensitivity,
            "limitation": "Targeted first-step density-gradient probe; not the historical failed batch, full actor loss, or native CUDA arithmetic."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.set_defaults(seed=1)
    parser.set_defaults(latent_seed=1)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {"schema": "docking_explicit_failure_audit_v1", "history": histories(args.manifest),
              "checkpoint_probe": checkpoint_probe(args.checkpoint, args.seed, args.latent_seed, args.episodes, args.samples),
              "source_sha256": evaluator.launcher.source_hashes(),
              "audit_sha256": digest(Path(__file__))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["checkpoint_probe"], indent=2))


if __name__ == "__main__":
    main()
