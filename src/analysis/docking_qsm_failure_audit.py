"""Recount QSM docking outcomes and trace a frozen sampler without training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/docking_full"
sys.path.insert(0, str(SOURCE))
from qsm_actor import qsm_actor, qsm_actor_loss, parent_critic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def trace(actor, obs, seed):
    torch.manual_seed(seed)
    x = torch.randn(len(obs), actor.act_dim)
    stages = []
    for t in range(actor.T - 1, -1, -1):
        eps = actor.eps(obs, x, torch.full((len(obs), 1), float(t)))
        coefficient = (1 - actor.alphas[t]) / torch.sqrt(1 - actor.alpha_hats[t])
        mean = (x - coefficient * eps) / torch.sqrt(actor.alphas[t])
        correction = coefficient * eps / torch.sqrt(actor.alphas[t])
        noise = (torch.sqrt(actor.betas[t]) * actor.ddpm_temperature * torch.randn_like(x)
                 if t > 0 else torch.zeros_like(x))
        preclip = mean + noise
        stages.append({"time_index": t, "eps_abs_median": eps.abs().median().item(),
                       "learned_correction_abs_median": correction.abs().median().item(),
                       "noise_abs_median": noise.abs().median().item(),
                       "preclip_abs_max": preclip.abs().max().item(),
                       "preclip_both_outside_fraction": (preclip.abs() > 1).all(1).float().mean().item(),
                       "preclip_any_outside_fraction": (preclip.abs() > 1).any(1).float().mean().item(),
                       "all_finite": bool(torch.isfinite(preclip).all())})
        x = preclip.clamp(-1, 1) if actor.clip_sampler else preclip
    x = x.clamp(-1, 1)
    torch.manual_seed(seed)
    native = actor.sample(obs)
    assert torch.equal(x, native), "Instrumented sampler differs from native sampler"
    return x, stages


def collect(ideal_path, checkpoint_path, evaluations, seed):
    ideal = json.loads(ideal_path.read_text())
    assert digest(checkpoint_path) == ideal["checkpoint_sha256"]
    sources = {str(p): digest(p) for p in (ideal_path, checkpoint_path)}
    for name, expected in ideal["source_sha256"].items():
        assert digest(SOURCE / name) == expected
        sources[str((SOURCE / name).relative_to(ROOT))] = expected
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = qsm_actor(5, 2, [-1, -1], [1, 1], **checkpoint["actor_constructor"])
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    goals = []
    for goal in ideal["goal_results"]:
        obs = torch.tensor(goal["observation"]).repeat(ideal["samples_per_goal"], 1)
        actions, stages = trace(actor, obs, goal["sampler_seed"])
        reference = torch.tensor(goal["actions"])
        difference = (actions-reference).abs().max().item()
        assert difference < 1e-5
        distances = (actions - torch.tensor(goal["target"])).norm(dim=1)
        goals.append({"goal": goal["goal"], "samples": len(obs), "stages": stages,
                      "reference_max_abs_difference": difference,
                      "both_exact_boundary_fraction": (actions.abs() == 1).all(1).float().mean().item(),
                      "any_exact_boundary_fraction": (actions.abs() == 1).any(1).float().mean().item(),
                      "min_target_distance": distances.min().item(),
                      "success_count": int((distances <= ideal["docking_radius"]).sum())})
    rows = []
    for path in evaluations:
        result = json.loads(path.read_text())
        episodes = result["episodes"]
        arrivals = [e for e in episodes if e["docking_distance"] is not None]
        successes = sum(e["success"] for e in episodes)
        assert episodes
        assert len(arrivals)/len(episodes) == result["metrics"]["arrival_rate"]
        assert all(e["success"] == int(e["docking_distance"] <= 2*result["sigma"]) for e in arrivals)
        rows.append({"seed": result["seed"], "step": result["checkpoint_global_step"],
                     "episodes": len(episodes), "arrivals": len(arrivals), "successes": successes,
                     "min_arrival_docking_distance": min((e["docking_distance"] for e in arrivals), default=None)})
        sources[str(path)] = digest(path)
    critics = [parent_critic(5, 2) for _ in range(2)]
    for critic, key in zip(critics, ("q1", "q2")):
        critic.load_state_dict(checkpoint[key], strict=True)
    torch.manual_seed(seed)
    obs = torch.tensor([g["observation"] for g in ideal["goal_results"]]).repeat(8, 1)
    actions = torch.rand(len(obs), 2)*2-1
    loss, gradient, eps = qsm_actor_loss(actor, *critics, obs, actions)
    loss.backward()
    actor_gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in actor_gradients)
    assert all(p.grad is None for c in critics for p in c.parameters())
    return {"schema": "docking_qsm_failure_audit_v1", "torch_version": torch.__version__,
            "device": "cpu", "sources": sources, "checkpoint_step": checkpoint["global_step"],
            "checkpoint_n_updates": checkpoint["n_updates"], "sampler_goals": goals,
            "narrow_navigation_loose_docking_recount": rows,
            "local_loss_check": {"synthetic_batch_size": len(obs), "seed": seed,
                                 "loss": loss.item(), "actor_gradients_finite": True,
                                 "critic_parameter_gradients_absent": True},
            "audit_sha256": digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ideal", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.set_defaults(seed=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = collect(args.ideal, args.checkpoint, args.evaluations, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
